#include "rr/learning_v2/training_logger.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>

#include "rr/domain/candidate.hpp"
#include "rr/learning_v2/training_log_schema.hpp"

namespace rr {

namespace {

// Fixed-precision, classic-locale double formatting — the SAME style as results_writer.cpp so the
// training-log CSVs are byte-identical regardless of the ambient LC_NUMERIC (D8). Six decimals
// matches the deterministic-metric CSVs; every score / similarity / feature / watch value uses it.
std::string num(double v, int precision = 6) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

// snake_case name for a retrieval source, matching serialization.cpp's canonical strings. Kept
// local (that function is file-private) since retrieval_sources is a NEW column A owns end-to-end.
const char *sourceName(CandidateSource source) {
    switch (source) {
    case CandidateSource::VectorHNSW:
        return "vector_hnsw";
    case CandidateSource::VectorExact:
        return "vector_exact";
    case CandidateSource::Popular:
        return "popular";
    case CandidateSource::Trending:
        return "trending";
    case CandidateSource::Fresh:
        return "fresh";
    case CandidateSource::CreatorAffinity:
        return "creator_affinity";
    case CandidateSource::Exploration:
        return "exploration";
    }
    return "vector_hnsw";
}

// The FULL merged retrieval-source union, pipe-joined in first-seen (merge) order, e.g.
// "vector_hnsw|trending". Pipe is CSV-safe (never a comma). The representative single label is
// derivable (exploration if present, else the first token) — package B reads whichever it needs.
std::string sourcesJoined(const std::vector<CandidateSource> &sources) {
    std::string out;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        if (i != 0) {
            out += '|';
        }
        out += sourceName(sources[i]);
    }
    return out;
}

// Build a JSON array from a column-name allowlist (std::string_view -> string).
template <std::size_t N> nlohmann::json columnArray(const std::array<std::string_view, N> &cols) {
    nlohmann::json a = nlohmann::json::array();
    for (const std::string_view c : cols) {
        a.push_back(std::string(c));
    }
    return a;
}

// Comma-join a column allowlist into a CSV header row (single source of truth: the frozen schema).
template <std::size_t N> std::string headerRow(const std::array<std::string_view, N> &cols) {
    std::string out;
    for (std::size_t i = 0; i < N; ++i) {
        if (i != 0) {
            out += ',';
        }
        out += std::string(cols[i]);
    }
    return out;
}

// candidates.csv header = the frozen prefix followed IN ORDER by the frozen feature columns.
std::string candidatesHeaderRow() {
    std::string out = headerRow(learning_v2::kCandidatesPrefixColumns);
    for (const std::string_view c : learning_v2::kFeatureColumns) {
        out += ',';
        out += std::string(c);
    }
    return out;
}

// "candidates-part0000.csv" style rotation-part name (contracts §2 / schema header).
std::string partFileName(const std::string &base, std::uint32_t index) {
    std::ostringstream oss;
    oss << base << "-part" << std::setw(4) << std::setfill('0') << index << ".csv";
    return oss.str();
}

// Flatten the 21 FeatureVector fields into the in-memory matrix row, in the SAME kFeatureColumns
// DECLARATION order writeFeatures() below emits and learned_ranker.cpp's featuresToRow() consumes —
// the frozen serving-purity ordering (contracts §2/§3). Kept adjacent to writeFeatures so the two
// can never drift.
std::array<float, learning_v2::kNumFeatures> featuresToArray(const FeatureVector &f) {
    return {
        f.similarity,         f.sessionTopic,    f.quality,
        f.freshness,          f.popularity,      f.trending,
        f.creatorAffinity,    f.exploration,     f.durationMatch,
        f.repetition,         f.impressionCount, f.visualMatch,
        f.musicMatch,         f.emotionalMatch,  f.clickbait,
        f.emotionalIntensity, f.usefulness,      f.productionQuality,
        f.informationDensity, f.languageMatch,   f.savePopularity,
    };
}

// Stream the 21 FeatureVector fields in kFeatureColumns DECLARATION ORDER (schema-authoritative).
void writeFeatures(std::ofstream &out, const FeatureVector &f) {
    out << ',' << num(f.similarity) << ',' << num(f.sessionTopic) << ',' << num(f.quality) << ','
        << num(f.freshness) << ',' << num(f.popularity) << ',' << num(f.trending) << ','
        << num(f.creatorAffinity) << ',' << num(f.exploration) << ',' << num(f.durationMatch) << ','
        << num(f.repetition) << ',' << num(f.impressionCount) << ',' << num(f.visualMatch) << ','
        << num(f.musicMatch) << ',' << num(f.emotionalMatch) << ',' << num(f.clickbait) << ','
        << num(f.emotionalIntensity) << ',' << num(f.usefulness) << ',' << num(f.productionQuality)
        << ',' << num(f.informationDensity) << ',' << num(f.languageMatch) << ','
        << num(f.savePopularity);
}

} // namespace

TrainingLogger::TrainingLogger(const LearningV2Config &config, std::filesystem::path runDir)
    : config_(config), trainingLogDir_(std::move(runDir) / "training_log"),
      keepMatrix_(config.learnedRanker) {}

void TrainingLogger::ensureDir() { std::filesystem::create_directories(trainingLogDir_); }

void TrainingLogger::openRequests() {
    ensureDir();
    requestsOut_.open(trainingLogDir_ / "requests.csv");
    requestsOut_ << headerRow(learning_v2::kRequestsColumns) << '\n';
    requestsOpened_ = true;
}

void TrainingLogger::openCandidatesPart(std::uint32_t index) {
    ensureDir();
    if (candidatesOut_.is_open()) {
        candidatesOut_.close();
    }
    candidatesOut_.open(trainingLogDir_ / partFileName("candidates", index));
    candidatesOut_ << candidatesHeaderRow() << '\n';
    candidatesRowsInPart_ = 0;
    candidatesPartIndex_ = index;
    candidatesOpened_ = true;
}

void TrainingLogger::openOutcomesPart(std::uint32_t index) {
    ensureDir();
    if (outcomesOut_.is_open()) {
        outcomesOut_.close();
    }
    outcomesOut_.open(trainingLogDir_ / partFileName("outcomes", index));
    outcomesOut_ << headerRow(learning_v2::kOutcomesColumns) << '\n';
    outcomesRowsInPart_ = 0;
    outcomesPartIndex_ = index;
    outcomesOpened_ = true;
}

// P22-HOOK(ranking). Contracts §2: emit the requests.csv row for a sampled request (shown-sample
// UNION pool-sample) and the matching candidates rows — the FULL pool for a pool-sampled request
// (position = feed slot for shown items, -1 for pool-only), else just the shown items.
void TrainingLogger::onRequestRanked(std::uint64_t requestId, const RecommendationRequest &request,
                                     const User &user, double effectiveEpsilon,
                                     const RankingCapture &capture,
                                     const std::vector<RankedReel> &rankedFeed) {
    // Pinned, rng-free sampling (contracts §1): the two salts drive independent decisions. Both
    // hooks recompute this from the request id, so no per-request state is carried across the run.
    const bool shownSel = learning_v2::logSampleSelected(requestId, learning_v2::kLogSampleSalt,
                                                         config_.logSampleRate);
    const bool poolSel = learning_v2::logSampleSelected(requestId, learning_v2::kLogPoolSampleSalt,
                                                        config_.logPoolSampleRate);
    if (!shownSel && !poolSel) {
        return;
    }

    // feed slot lookup by reel id (positions come from the shown feed, not the pool order).
    std::unordered_map<std::uint32_t, std::size_t> feedPos;
    feedPos.reserve(rankedFeed.size());
    for (const RankedReel &r : rankedFeed) {
        feedPos.emplace(r.reelId.value, r.rank);
    }

    if (!requestsOpened_) {
        openRequests();
    }
    requestsOut_ << requestId << ',' << user.id.value << ',' << request.sessionId.value << ','
                 << request.requestTime << ',' << request.feedSize << ',' << num(effectiveEpsilon)
                 << ',' << capture.rows.size() << ',' << rankedFeed.size() << ','
                 << (poolSel ? 1 : 0) << '\n';

    for (const RankingCaptureRow &row : capture.rows) {
        const auto it = feedPos.find(row.reelId.value);
        const bool isShown = it != feedPos.end();
        // Shown-only sample: emit just the shown impressions. Pool sample: emit the whole pool.
        if (!poolSel && !isShown) {
            continue;
        }
        const long position = isShown ? static_cast<long>(it->second) : -1L;

        // Phase 23 (contracts §3): stash the served features for a SHOWN candidate of a
        // SHOWN-sampled request into the in-memory matrix, keyed (request_id, reel_id). Only
        // shown-sampled rows are matrixed — those are exactly the ones whose outcomes arrive at
        // onImpressionOutcome (same pinned predicate), so the join is complete. No-op unless the
        // learned-ranker matrix is kept.
        if (keepMatrix_ && shownSel && isShown) {
            learning_v2::ShownFeatureRow &m = matrix_[{requestId, row.reelId.value}];
            m.features = featuresToArray(row.features);
            m.hasFeatures = true;
        }

        if (!candidatesOpened_) {
            openCandidatesPart(0);
        } else if (candidatesRowsInPart_ >= config_.logMaxRowsPerFile) {
            openCandidatesPart(candidatesPartIndex_ + 1); // rotate (contracts §2)
        }
        candidatesOut_ << requestId << ',' << row.reelId.value << ',' << row.poolRank << ','
                       << (isShown ? 1 : 0) << ',' << position << ',' << num(row.servedScore) << ','
                       << (row.explorationLabeled ? 1 : 0) << ',' << sourcesJoined(row.sources)
                       << ',' << num(row.retrievalSimilarity);
        writeFeatures(candidatesOut_, row.features);
        candidatesOut_ << '\n';
        ++candidatesRowsInPart_;
    }
}

// P22-HOOK(outcome). Contracts §2: the SEPARATE label table (evaluation-side observables only).
// completed/liked/shared/followed/not_interested come from `outcome` (the event's single `type` is
// lossy for simultaneous signals); the watch / V2-engagement / exit observables come from `event`.
void TrainingLogger::onImpressionOutcome(const InteractionEvent &event,
                                         const BehaviourOutcome &outcome) {
    if (!learning_v2::logSampleSelected(event.requestId, learning_v2::kLogSampleSalt,
                                        config_.logSampleRate)) {
        return; // outcomes join to SHOWN-sampled impressions only.
    }

    // Phase 23 (contracts §3): join the observable outcome labels to the matrix row whose features
    // were captured at ranking. The SIX §4.21 targets only: watch_ratio + shared/followed/
    // not_interested (from `outcome`, the event's single `type` being lossy) + session_exit (=
    // event.observedExitAfterImpression). find() (not operator[]) so an unmatched outcome — a shown
    // impression whose features weren't captured, which does not happen for a shown-sampled request
    // — creates no feature-less row. The first outcome flips hasOutcome and counts the row as
    // trainable.
    if (keepMatrix_) {
        auto mit = matrix_.find({event.requestId, event.reelId.value});
        if (mit != matrix_.end()) {
            learning_v2::ShownFeatureRow &m = mit->second;
            m.watchRatio = static_cast<float>(event.watchRatio);
            m.shared = outcome.shared ? 1 : 0;
            m.followed = outcome.followed ? 1 : 0;
            m.notInterested = outcome.notInterested ? 1 : 0;
            m.sessionExit = event.observedExitAfterImpression ? 1 : 0;
            if (m.hasFeatures && !m.hasOutcome) {
                ++matrixCompleteRows_;
            }
            m.hasOutcome = true;
        }
    }

    if (!outcomesOpened_) {
        openOutcomesPart(0);
    } else if (outcomesRowsInPart_ >= config_.logMaxRowsPerFile) {
        openOutcomesPart(outcomesPartIndex_ + 1); // rotate (contracts §2)
    }
    outcomesOut_ << event.requestId << ',' << event.reelId.value << ',' << event.positionInFeed
                 << ',' << num(event.watchSeconds) << ',' << num(event.watchRatio) << ','
                 << (outcome.completed ? 1 : 0) << ',' << (outcome.liked ? 1 : 0) << ','
                 << (outcome.shared ? 1 : 0) << ',' << (outcome.followed ? 1 : 0) << ','
                 << (outcome.notInterested ? 1 : 0) << ',' << (event.commented ? 1 : 0) << ','
                 << (event.saved ? 1 : 0) << ',' << (event.profileVisited ? 1 : 0) << ','
                 << (event.observedExitAfterImpression ? 1 : 0) << '\n';
    ++outcomesRowsInPart_;
}

// Phase 23 (contracts §2/§3). Join the OBSERVABLE survey likert to its matrix row. The likert is an
// observable explicit-feedback label (the recommender-visible survey response); the hidden
// immediateSatisfaction it was derived from never reaches this NON-carve-out module (D18). No-op
// unless the matrix is kept and the row exists (features captured at ranking).
void TrainingLogger::onSurvey(std::uint64_t requestId, ReelId reelId, int likert) {
    if (!keepMatrix_) {
        return;
    }
    auto mit = matrix_.find({requestId, reelId.value});
    if (mit != matrix_.end()) {
        mit->second.likert = static_cast<std::uint8_t>(likert);
    }
}

// Phase 23 (contracts §3). Snapshot the TRAINABLE rows (features captured AND outcome joined) in
// the map's canonical (request_id, reel_id) order — std::map iteration is ordered, so this is
// hash-map-order-independent and bit-reproducible across platforms (the retraining-determinism
// contract, §5). Called only at retrain boundaries.
std::vector<learning_v2::ShownFeatureRow> TrainingLogger::snapshotMatrix() const {
    std::vector<learning_v2::ShownFeatureRow> rows;
    rows.reserve(matrixCompleteRows_);
    for (const auto &[key, row] : matrix_) {
        if (row.hasFeatures && row.hasOutcome) {
            rows.push_back(row);
        }
    }
    return rows;
}

// P22-HOOK(finish). Flush/close the streams, ensure every §2 table exists (header-only when a run
// logged none, so package B's reader always finds them), and write the frozen schema descriptor.
void TrainingLogger::finish() {
    ensureDir();
    if (!requestsOpened_) {
        openRequests();
    }
    if (!candidatesOpened_) {
        openCandidatesPart(0);
    }
    if (!outcomesOpened_) {
        openOutcomesPart(0);
    }
    requestsOut_.flush();
    candidatesOut_.flush();
    outcomesOut_.flush();

    nlohmann::json candidates = columnArray(learning_v2::kCandidatesPrefixColumns);
    for (const std::string_view c : learning_v2::kFeatureColumns) {
        candidates.push_back(std::string(c)); // full candidates.csv header = prefix ++ features
    }

    nlohmann::json schema;
    schema["schema_version"] = learning_v2::kSchemaVersion;
    schema["tables"] = {
        {"requests", columnArray(learning_v2::kRequestsColumns)},
        {"candidates", candidates},
        {"outcomes", columnArray(learning_v2::kOutcomesColumns)},
        {"survey", columnArray(learning_v2::kSurveyColumns)},
    };
    schema["feature_columns"] = columnArray(learning_v2::kFeatureColumns);
    schema["sampling"] = {
        {"predicate", "pinnedHash01(request_id ^ salt) < rate"},
        {"log_sample_salt", learning_v2::kLogSampleSalt},
        {"log_pool_sample_salt", learning_v2::kLogPoolSampleSalt},
    };
    // File layout so package B's reader can glob deterministically (contracts §2 rotation parts).
    schema["file_layout"] = {
        {"requests", "requests.csv"},
        {"candidates", "candidates-partNNNN.csv"},
        {"outcomes", "outcomes-partNNNN.csv"},
        {"survey", "survey.csv"},
        {"part_start_index", 0},
        {"max_rows_per_part", config_.logMaxRowsPerFile},
    };
    schema["config"] = config_; // learning_v2 echo (training_log/log_sample_rate/.../survey)
    // The one hidden-derived table is clearly labeled (contracts §2/§4b): the purity audit exempts
    // survey.csv's documented columns from the forbidden-substring scan.
    schema["hidden_derived_tables"] = nlohmann::json::array({"survey"});

    std::ofstream out(trainingLogDir_ / "schema.json");
    out << schema.dump(2) << '\n';
}

} // namespace rr
