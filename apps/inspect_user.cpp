// inspect_user — dumps topic/reel/creator/user distribution summaries as JSON (Phase 2, task 7),
// with --explain-user, a ranked hnsw_ranker feed for one user with the per-feature contribution
// breakdown of every item (Phase 6, task 3 / TDD 14.4), and with --v2-summary, the Realism V2
// content/user factor-model distributions (Phase 13, plan task 8).
//
// A first-cut eyeballing tool: generates a full synthetic dataset via rr::generateDataset and
// reports a nearest-topic histogram (for reels and for users, by nearest topic centre) and a
// quality histogram (reel intrinsicQuality, creator baseQuality), plus basic scale counts. This is
// an inspection tool, not a benchmark (no timing claims) — see apps/benchmark_retrieval.cpp for
// that.
//
// --v2-summary (Phase 13, D18 evaluation carve-out): this app is simulator-side tooling, not a
// recommender/ranking codepath, so it may read hidden state for inspection — the same carve-out
// benchmark_recommender uses (Phase 11 precedent). It reports archetype identity and hidden trait
// values that must NEVER reach recommender-visible state anywhere else; every field printed here
// is explicitly labeled as simulator-only inspection output. Robust to gate-off / A-B-stub state:
// empty hiddenReelStates yields an empty archetype table, and empty/zero modality embeddings are
// counted and reported rather than crashing.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rr/core/embedding.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/infrastructure/archetype_config.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/hidden/hidden_reel_state.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/simulator.hpp"

namespace {

// Index of the topic whose centre is closest (max dot product; embeddings are unit vectors so
// this ranks identically to nearest-by-Euclidean-distance, design decision D3) to `e`.
size_t nearestTopic(const rr::Embedding &e, const std::vector<rr::Topic> &topics) {
    size_t best = 0;
    float bestDot = -2.0f;
    for (size_t i = 0; i < topics.size(); ++i) {
        const float d = rr::dot(e, topics[i].centre);
        if (d > bestDot) {
            bestDot = d;
            best = i;
        }
    }
    return best;
}

std::vector<uint64_t> histogram(const std::vector<float> &values, int numBins) {
    std::vector<uint64_t> bins(static_cast<size_t>(numBins), 0);
    for (float v : values) {
        int bin = std::clamp(static_cast<int>(v * static_cast<float>(numBins)), 0, numBins - 1);
        ++bins[static_cast<size_t>(bin)];
    }
    return bins;
}

// Writes `out` exactly the way every existing mode does: pretty-printed to stdout, or to a file
// (creating parent directories) with a confirmation on stderr, when --out is given. Factored out
// so --v2-summary shares the identical, already-proven behavior instead of a third copy of it.
void writeOutput(const nlohmann::json &out, const std::string &outPath) {
    if (outPath.empty()) {
        std::cout << out.dump(2) << "\n";
        return;
    }
    std::filesystem::path p(outPath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream f(p);
    f << out.dump(2) << "\n";
    std::cerr << "wrote " << p.string() << "\n";
}

// --- --v2-summary helpers (Phase 13, plan task 8) --------------------------------------------

// Tally of how many embeddings in a modality-embedding collection are unit-norm vs. empty
// (the gate-off / stub-A/B default: Embedding{} default-constructs to size 0, per D5's
// `using Embedding = std::vector<float>`) vs. populated-but-not-unit-length (would indicate a
// bug once the real V2 generators are live, since D3/D5 mandate every embedding be L2-normalized
// at creation — reported distinctly rather than silently folded into either other bucket).
struct NormTally {
    uint64_t total = 0;
    uint64_t unitNorm = 0;
    uint64_t empty = 0;
    uint64_t nonUnitPopulated = 0;
};

template <typename Container, typename Proj>
NormTally tallyNormsBy(const Container &items, Proj &&proj) {
    NormTally t;
    for (const auto &item : items) {
        const rr::Embedding &e = proj(item);
        ++t.total;
        if (e.empty()) {
            ++t.empty;
        } else if (rr::isValid(e)) {
            ++t.unitNorm;
        } else {
            ++t.nonUnitPopulated;
        }
    }
    return t;
}

nlohmann::json normTallyJson(const NormTally &t) {
    return {{"total", t.total},
            {"unit_norm", t.unitNorm},
            {"empty", t.empty},
            {"non_unit_populated", t.nonUnitPopulated}};
}

// mean/min/max of one scalar field across a collection (typically HiddenUserState); count == 0
// yields all-zero rather than NaN/±inf so it always serializes cleanly (e.g. a --v2-summary run
// against a zero-user dataset must not crash or emit invalid JSON).
struct ScalarStats {
    uint64_t count = 0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
};

template <typename Container, typename Proj>
ScalarStats scalarStatsBy(const Container &items, Proj &&proj) {
    ScalarStats s;
    if (items.empty()) {
        return s;
    }
    double sum = 0.0;
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (const auto &item : items) {
        const float v = proj(item);
        sum += v;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    s.count = items.size();
    s.mean = sum / static_cast<double>(items.size());
    s.min = lo;
    s.max = hi;
    return s;
}

nlohmann::json scalarStatsJson(const ScalarStats &s) {
    return {{"count", s.count}, {"mean", s.mean}, {"min", s.min}, {"max", s.max}};
}

} // namespace

// --explain-user mode (Phase 6, task 3): cold-start the population, warm the simulation up with
// `warmupRounds` uniformly-random impressions per user (so popularity/trending counters and the
// per-user creator-affinity/duration signals are non-trivial), then serve ONE hnsw_ranker feed
// for the chosen user and dump it in the TDD 14.4 explanation shape. Deterministic for a given
// (config, seed): warm-up picks come from the dedicated "inspect_warmup" stream (D8).
nlohmann::json explainUser(const rr::ExperimentConfig &config, uint64_t seed, uint32_t userIdArg,
                           uint32_t warmupRounds) {
    rr::GeneratedDataset ds = rr::generateDataset(config.simulation, seed);
    if (userIdArg >= ds.users.size()) {
        std::cerr << "--explain-user " << userIdArg << " out of range (users: " << ds.users.size()
                  << ")\n";
        std::exit(2);
    }
    rr::applyColdStart(ds.users, rr::globalAveragePreference(ds.hiddenStates));

    rr::Simulator sim(config.behaviour, config.reward, rr::forkRng(seed, "behaviour"),
                      config.learning.recentWindow, config.ranking.trendingHalfLifeSeconds);
    rr::Rng warmupRng = rr::forkRng(seed, "inspect_warmup");
    for (uint32_t round = 0; round < warmupRounds; ++round) {
        for (rr::User &user : ds.users) {
            const uint32_t reelIdx = warmupRng.uniformInt(static_cast<uint32_t>(ds.reels.size()));
            rr::Reel &reel = ds.reels[reelIdx];
            if (!reel.active) {
                continue;
            }
            sim.step(user, ds.hiddenStates[user.id.value], reel, ds.creators[reel.creatorId.value]);
        }
    }

    rr::RecommenderDeps deps{ds.reels, ds.users, config};
    auto rec = rr::makeRecommender(rr::RecommendationAlgorithm::HnswRanker, deps,
                                   rr::forkRng(seed, "recommender"));

    rr::RecommendationRequest req{};
    req.userId = rr::UserId{userIdArg};
    req.feedSize = config.recommendation.feedSize;
    req.candidateLimit = config.recommendation.vectorCandidates;
    req.requestTime = sim.now();
    const rr::RecommendationResponse response = rec->recommend(req);

    nlohmann::json feed = nlohmann::json::array();
    for (const rr::RankedReel &item : response.reels) {
        const rr::Reel &reel = ds.reels[item.reelId.value];
        nlohmann::json contributions;
        for (const auto &[key, value] : item.featureContributions) {
            contributions[key] = value;
        }
        nlohmann::json sources = nlohmann::json::array();
        for (rr::CandidateSource s : item.sources) {
            sources.push_back(static_cast<int>(s));
        }
        feed.push_back({{"rank", item.rank},
                        {"reel_id", item.reelId.value},
                        {"creator_id", reel.creatorId.value},
                        {"primary_topic", reel.primaryTopic.value},
                        {"score", item.score},
                        {"sources", sources},
                        {"contributions", contributions}});
    }

    return nlohmann::json{{"seed", seed},
                          {"user_id", userIdArg},
                          {"warmup_rounds", warmupRounds},
                          {"request_time", req.requestTime},
                          {"algorithm", "hnsw_ranker"},
                          {"feed", feed}};
}

// --v2-summary mode (Phase 13, plan task 8): dumps the Realism V2 content/user factor-model
// distributions for dataset inspection. D18 evaluation carve-out (see file header) — this is
// simulator-side tooling, never a recommender codepath, so it may read archetype identity and
// hidden trait values directly. Every value here is generated by rr::generateDataset alone (no
// simulation/behaviour step, unlike --explain-user): with realism.content_v2 off, or with either
// sibling package's augmenter still a no-op stub, hiddenReelStates is empty and every V2 field
// keeps its gate-off default (D17) — the function must produce a valid, non-crashing report in
// that shape, which is exactly what the per-branch guards below do.
nlohmann::json v2Summary(const rr::ExperimentConfig &experiment, uint64_t seed) {
    const rr::GeneratedDataset ds =
        rr::generateDataset(experiment.simulation, experiment.realism, seed);

    // Per-archetype reel counts + per-archetype means of the eight V2 Reel scalars, joining
    // reels[i] with the index-aligned hiddenReelStates[i] (V2 TDD 5). hiddenReelStates is empty
    // under gate-off/stub state (D17) — and, defensively, any size mismatch also yields an empty
    // table rather than an out-of-bounds join.
    nlohmann::json archetypes = nlohmann::json::object();
    const bool archetypesJoinable =
        !ds.hiddenReelStates.empty() && ds.hiddenReelStates.size() == ds.reels.size();
    if (archetypesJoinable) {
        struct Accum {
            uint64_t count = 0;
            double usefulness = 0.0;
            double humour = 0.0;
            double novelty = 0.0;
            double productionQuality = 0.0;
            double controversy = 0.0;
            double clickbaitStrength = 0.0;
            double informationDensity = 0.0;
            double emotionalIntensity = 0.0;
        };
        std::map<std::string, Accum> byArchetype;
        for (size_t i = 0; i < ds.reels.size(); ++i) {
            const rr::Reel &reel = ds.reels[i];
            const rr::HiddenReelState &hidden = ds.hiddenReelStates[i];
            const std::string name =
                hidden.archetypeIndex < experiment.realism.archetypes.size()
                    ? experiment.realism.archetypes[hidden.archetypeIndex].name
                    : ("unknown_archetype_index_" + std::to_string(hidden.archetypeIndex));
            Accum &a = byArchetype[name];
            ++a.count;
            a.usefulness += reel.usefulness;
            a.humour += reel.humour;
            a.novelty += reel.novelty;
            a.productionQuality += reel.productionQuality;
            a.controversy += reel.controversy;
            a.clickbaitStrength += reel.clickbaitStrength;
            a.informationDensity += reel.informationDensity;
            a.emotionalIntensity += reel.emotionalIntensity;
        }
        for (const auto &[name, a] : byArchetype) {
            const double n = static_cast<double>(a.count);
            archetypes[name] = {{"count", a.count},
                                {"mean_usefulness", a.usefulness / n},
                                {"mean_humour", a.humour / n},
                                {"mean_novelty", a.novelty / n},
                                {"mean_production_quality", a.productionQuality / n},
                                {"mean_controversy", a.controversy / n},
                                {"mean_clickbait_strength", a.clickbaitStrength / n},
                                {"mean_information_density", a.informationDensity / n},
                                {"mean_emotional_intensity", a.emotionalIntensity / n}};
        }
    }

    // Language histograms keyed by the OBSERVED language id (not by config.realism.languages),
    // so gate-off's all-default-LanguageId{0} dataset reports cleanly as a single bin.
    std::map<uint32_t, uint64_t> reelLangHist;
    for (const auto &r : ds.reels) {
        ++reelLangHist[r.language.value];
    }
    std::map<uint32_t, uint64_t> userLangHist;
    for (const auto &h : ds.hiddenStates) {
        ++userLangHist[h.primaryLanguage.value];
    }
    nlohmann::json reelLangJson = nlohmann::json::object();
    for (const auto &[id, count] : reelLangHist) {
        reelLangJson[std::to_string(id)] = count;
    }
    nlohmann::json userLangJson = nlohmann::json::object();
    for (const auto &[id, count] : userLangHist) {
        userLangJson[std::to_string(id)] = count;
    }

    // Per-scalar hidden user preference/susceptibility/forward-trait summary (V2 TDD 4.2 + 5):
    // every V2 scalar field HiddenUserState carries, mean/min/max over the population.
    nlohmann::json prefSummary;
    prefSummary["usefulness_preference"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.usefulnessPreference; }));
    prefSummary["humour_preference"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.humourPreference; }));
    prefSummary["controversy_tolerance"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.controversyTolerance; }));
    prefSummary["novelty_seeking"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.noveltySeeking; }));
    prefSummary["clickbait_susceptibility"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.clickbaitSusceptibility; }));
    prefSummary["information_tolerance"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.informationTolerance; }));
    prefSummary["language_mismatch_tolerance"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.languageMismatchTolerance; }));
    prefSummary["repetition_tolerance"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.repetitionTolerance; }));
    prefSummary["novelty_tolerance"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.noveltyTolerance; }));
    prefSummary["creator_loyalty"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.creatorLoyalty; }));
    prefSummary["habit_strength"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.habitStrength; }));
    prefSummary["platform_trust"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.platformTrust; }));
    prefSummary["baseline_daily_usage"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.baselineDailyUsage; }));
    prefSummary["preference_plasticity"] = scalarStatsJson(scalarStatsBy(
        ds.hiddenStates, [](const rr::HiddenUserState &h) { return h.preferencePlasticity; }));

    // Modality embedding norm checks, both sides (V2 TDD 4.1/4.2): visual/music/emotional, reel
    // side and user (hidden preference) side.
    nlohmann::json normChecks;
    normChecks["reel"] = {
        {"visual", normTallyJson(tallyNormsBy(ds.reels,
                                              [](const rr::Reel &r) -> const rr::Embedding & {
                                                  return r.visualStyleEmbedding;
                                              }))},
        {"music", normTallyJson(tallyNormsBy(ds.reels,
                                             [](const rr::Reel &r) -> const rr::Embedding & {
                                                 return r.musicEmbedding;
                                             }))},
        {"emotional", normTallyJson(tallyNormsBy(ds.reels,
                                                 [](const rr::Reel &r) -> const rr::Embedding & {
                                                     return r.emotionalToneEmbedding;
                                                 }))},
    };
    normChecks["user"] = {
        {"visual",
         normTallyJson(tallyNormsBy(ds.hiddenStates,
                                    [](const rr::HiddenUserState &h) -> const rr::Embedding & {
                                        return h.visualPreference;
                                    }))},
        {"music",
         normTallyJson(tallyNormsBy(ds.hiddenStates,
                                    [](const rr::HiddenUserState &h) -> const rr::Embedding & {
                                        return h.musicPreference;
                                    }))},
        {"emotional",
         normTallyJson(tallyNormsBy(ds.hiddenStates,
                                    [](const rr::HiddenUserState &h) -> const rr::Embedding & {
                                        return h.emotionalPreference;
                                    }))},
    };

    nlohmann::json out;
    out["note"] = "simulator-side inspection (D18 evaluation carve-out): archetype identities "
                  "and hidden traits are printed here for dataset inspection only";
    out["seed"] = seed;
    out["gate"] = {{"content_v2", experiment.realism.contentV2}};
    out["counts"] = {{"reels", ds.reels.size()},
                     {"users", ds.users.size()},
                     {"hidden_reel_states", ds.hiddenReelStates.size()},
                     {"hidden_user_states", ds.hiddenStates.size()}};
    out["archetypes"] = archetypes;
    out["reel_language_histogram"] = reelLangJson;
    out["user_primary_language_histogram"] = userLangJson;
    out["user_preference_summary"] = prefSummary;
    out["modality_embedding_norms"] = normChecks;
    return out;
}

int main(int argc, char **argv) {
    std::string configPath;
    uint64_t seed = 42;
    std::string outPath;        // empty => stdout
    int64_t explainUserId = -1; // >= 0 => explanation mode
    uint32_t warmupRounds = 30;
    bool v2SummaryMode = false; // --v2-summary (Phase 13)

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << what << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--config") {
            configPath = next("--config");
        } else if (a == "--seed") {
            seed = std::stoull(next("--seed"));
        } else if (a == "--out") {
            outPath = next("--out");
        } else if (a == "--explain-user") {
            explainUserId = std::stoll(next("--explain-user"));
        } else if (a == "--warmup") {
            warmupRounds = static_cast<uint32_t>(std::stoul(next("--warmup")));
        } else if (a == "--v2-summary") {
            v2SummaryMode = true;
        } else if (a == "--help" || a == "-h") {
            std::cout << "usage: inspect_user [--config path] [--seed N] [--out path]\n"
                         "                    [--explain-user ID [--warmup ROUNDS]]\n"
                         "                    [--v2-summary]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    rr::ExperimentConfig experiment; // defaults: 10k users / 100k reels / 5k creators / 32 topics.
    if (!configPath.empty()) {
        experiment = rr::loadExperimentConfig(configPath);
    }
    const rr::SimulationConfig &config = experiment.simulation;

    if (explainUserId >= 0) {
        const nlohmann::json out =
            explainUser(experiment, seed, static_cast<uint32_t>(explainUserId), warmupRounds);
        writeOutput(out, outPath);
        return 0;
    }

    if (v2SummaryMode) {
        const nlohmann::json out = v2Summary(experiment, seed);
        writeOutput(out, outPath);
        return 0;
    }

    rr::GeneratedDataset dataset = rr::generateDataset(config, seed);

    std::vector<uint64_t> reelTopicHistogram(dataset.topics.size(), 0);
    std::vector<float> reelQualities;
    reelQualities.reserve(dataset.reels.size());
    for (const auto &r : dataset.reels) {
        ++reelTopicHistogram[nearestTopic(r.embedding, dataset.topics)];
        reelQualities.push_back(r.intrinsicQuality);
    }

    std::vector<uint64_t> userTopicHistogram(dataset.topics.size(), 0);
    for (const auto &h : dataset.hiddenStates) {
        ++userTopicHistogram[nearestTopic(h.hiddenPreference, dataset.topics)];
    }

    std::vector<float> creatorQualities;
    creatorQualities.reserve(dataset.creators.size());
    for (const auto &c : dataset.creators) {
        creatorQualities.push_back(c.baseQuality);
    }

    nlohmann::json out;
    out["seed"] = seed;
    out["counts"] = {{"topics", dataset.topics.size()},
                     {"creators", dataset.creators.size()},
                     {"reels", dataset.reels.size()},
                     {"users", dataset.users.size()}};
    out["reel_nearest_topic_histogram"] = reelTopicHistogram;
    out["user_nearest_topic_histogram"] = userTopicHistogram;
    out["reel_quality_histogram"] = histogram(reelQualities, 10);
    out["creator_quality_histogram"] = histogram(creatorQualities, 10);

    writeOutput(out, outPath);

    return 0;
}
