#include "rr/recommendation/orchestrator.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rr/candidate_sources/exploration_candidate_source.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/recommendation/reranker.hpp"
#include "rr/recommendation/seen_filter.hpp"

namespace rr {

namespace {

// A reel surviving the merge stage: its best (smallest) retrieval distance, the similarity paired
// with THAT distance, and the union of every source label that produced it.
struct MergedCandidate {
    ReelId reelId;
    float retrievalDistance;
    float retrievalSimilarity;
    std::vector<CandidateSource> sources;
};

// The strict weak ordering shared by the pool cap and the identity ranking stage: descending
// retrieval similarity, ties broken by ascending ReelId. ReelIds are unique after dedup, so this
// is a TOTAL order and every downstream selection/sort is deterministic.
bool moreRelevant(const MergedCandidate &a, const MergedCandidate &b) {
    if (a.retrievalSimilarity != b.retrievalSimilarity) {
        return a.retrievalSimilarity > b.retrievalSimilarity;
    }
    return a.reelId.value < b.reelId.value;
}

// The single source label carried on the Candidate handed to the ranker (and read by the
// FeatureExtractor's exploration feature and the guaranteed-slot rule). Exploration WINS the slot
// whenever it appears in the merged label set — a reel any exploration mode surfaced counts as
// exploration even if an exploitation source also produced it — otherwise the first-seen label is
// kept. Non-exploration recommenders never have Exploration in a label set, so this reduces to
// sources.front() and every pre-Phase-8 output stays byte-identical.
CandidateSource representativeSource(const std::vector<CandidateSource> &sources) {
    for (CandidateSource s : sources) {
        if (s == CandidateSource::Exploration) {
            return s;
        }
    }
    return sources.front();
}

bool isExplorationLabeled(const Candidate &c) { return c.source == CandidateSource::Exploration; }

// Phase 22 served-time ranking capture (contracts §7, see orchestrator.hpp). Runs the runner-
// supplied extractor ONCE on the best-first served pool and records one row per candidate in
// poolRank order. The extractor is pure over (reels, config, contentV2, user, pool, now) and every
// pool-relative feature is set-invariant, so `features[i]` is byte-identical to the FeatureVector
// the ranker computed for the same reel. `fullSources[i]` is the complete merged retrieval-source
// union for `bestFirstPool[i]` (Candidate carries only the single representative label). Called
// only when a sink + extractor were supplied (a sampled logging request); never on a
// production/golden path.
void fillRankingCapture(RankingCapture &cap, const User &user, Timestamp now,
                        const std::vector<Candidate> &bestFirstPool,
                        const std::vector<std::vector<CandidateSource>> &fullSources) {
    const std::vector<FeatureVector> features = cap.extractor->extract(user, bestFirstPool, now);
    cap.rows.clear();
    cap.rows.reserve(bestFirstPool.size());
    for (std::size_t i = 0; i < bestFirstPool.size(); ++i) {
        const Candidate &c = bestFirstPool[i];
        RankingCaptureRow row;
        row.reelId = c.reelId;
        row.poolRank = i;
        row.servedScore = c.rankingScore;
        row.explorationLabeled = isExplorationLabeled(c);
        row.retrievalSimilarity = c.retrievalSimilarity;
        row.sources = fullSources[i];
        row.features = features[i];
        cap.rows.push_back(std::move(row));
    }
}

} // namespace

Orchestrator::Orchestrator(std::vector<CandidateGenerator *> sources,
                           const std::vector<Reel> &reels, const Ranker *ranker,
                           const ExplorationCandidateSource *explorationSource,
                           uint32_t guaranteedSlots, const Reranker *reranker)
    : sources_(std::move(sources)), reels_(reels), ranker_(ranker),
      explorationSource_(explorationSource), guaranteedSlots_(guaranteedSlots),
      reranker_(reranker) {}

// Guaranteed-exploration-slot promotion (Phase 8, TDD 12.7 task 3). `ranked` is the full capped
// pool in best-first order; only the first `feedSize` survive truncation. We ensure that prefix
// carries g = min(lastFiredSlots, guaranteedSlots, exploration items in the pool) exploration-
// labeled items:
//   * P = the `need` HIGHEST-ranked exploration items strictly below the cut (smallest indices);
//   * E = the `need` LOWEST-ranked non-exploration items in the prefix (largest indices);
// the new feed is (prefix minus E, order preserved) followed by P (in rank order); the evicted and
// the un-promoted below-cut items form the tail in ascending original rank. Everything's relative
// order is otherwise preserved, so the reorder is fully deterministic. lastFiredSlots <= feedSize
// (the source draws exactly feedSize gates), which guarantees enough non-exploration items exist in
// the prefix to evict.
void Orchestrator::applyExplorationGuarantee(std::vector<Candidate> &ranked,
                                             std::size_t feedSize) const {
    if (explorationSource_ == nullptr || guaranteedSlots_ == 0) {
        return;
    }
    // When the whole pool already fits in the feed, every exploration item is already shown.
    if (ranked.size() <= feedSize) {
        return;
    }
    const std::size_t fired = explorationSource_->lastFiredSlots();
    if (fired == 0) {
        return;
    }

    std::size_t explTotal = 0;
    std::size_t inPrefix = 0;
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        if (isExplorationLabeled(ranked[i])) {
            ++explTotal;
            if (i < feedSize) {
                ++inPrefix;
            }
        }
    }
    const std::size_t g =
        std::min<std::size_t>({fired, static_cast<std::size_t>(guaranteedSlots_), explTotal});
    if (inPrefix >= g) {
        return;
    }
    const std::size_t need = g - inPrefix;

    // P: highest-ranked exploration items below the cut (there are always >= need of them, since
    // explTotal - inPrefix = below-cut exploration items and need = g - inPrefix <= that).
    std::vector<std::size_t> promote;
    promote.reserve(need);
    for (std::size_t i = feedSize; i < ranked.size() && promote.size() < need; ++i) {
        if (isExplorationLabeled(ranked[i])) {
            promote.push_back(i);
        }
    }
    // E: lowest-ranked non-exploration items in the prefix (walk the prefix from the back).
    std::vector<char> evicted(ranked.size(), 0);
    std::size_t evictedCount = 0;
    for (std::size_t i = feedSize; i-- > 0 && evictedCount < need;) {
        if (!isExplorationLabeled(ranked[i])) {
            evicted[i] = 1;
            ++evictedCount;
        }
    }
    std::vector<char> promoted(ranked.size(), 0);
    for (std::size_t i : promote) {
        promoted[i] = 1;
    }

    std::vector<Candidate> reordered;
    reordered.reserve(ranked.size());
    // Retained prefix (order preserved), then the promoted items (rank order)...
    for (std::size_t i = 0; i < feedSize; ++i) {
        if (!evicted[i]) {
            reordered.push_back(std::move(ranked[i]));
        }
    }
    for (std::size_t i : promote) {
        reordered.push_back(std::move(ranked[i]));
    }
    // ...then the tail: evicted prefix items and un-promoted below-cut items, ascending rank.
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        const bool alreadyUsed = (i < feedSize) ? (evicted[i] == 0) : (promoted[i] != 0);
        if (!alreadyUsed) {
            reordered.push_back(std::move(ranked[i]));
        }
    }
    ranked = std::move(reordered);
}

RecommendationResponse Orchestrator::recommend(const User &user,
                                               const RecommendationRequest &request) {
    Stopwatch total;
    RecommendationResponse response{};

    // --- Retrieval (TDD 13 step 1): run every source, accumulate raw candidates. Timed as a
    // single stage. candidatesRetrieved is the PRE-dedup sum over all sources.
    Stopwatch retrieval;
    std::vector<Candidate> raw;
    for (CandidateGenerator *source : sources_) {
        std::vector<Candidate> produced = source->generate(user, request);
        response.candidatesRetrieved += produced.size();
        raw.insert(raw.end(), std::make_move_iterator(produced.begin()),
                   std::make_move_iterator(produced.end()));
    }
    response.retrievalLatencyMs = retrieval.elapsedMs();

    // --- Merge + dedup by reel id (TDD 13 steps 2-3), preserving ALL source labels. Merge rule:
    // on a duplicate keep the SMALLEST retrieval distance (and the similarity paired with it),
    // and union the source labels in first-seen order. The final ordering is imposed by
    // moreRelevant() below, so the result is independent of hash-map iteration order.
    std::unordered_map<ReelId, MergedCandidate> merged;
    merged.reserve(raw.size());
    for (const Candidate &c : raw) {
        auto it = merged.find(c.reelId);
        if (it == merged.end()) {
            merged.emplace(
                c.reelId,
                MergedCandidate{c.reelId, c.retrievalDistance, c.retrievalSimilarity, {c.source}});
        } else {
            MergedCandidate &m = it->second;
            if (c.retrievalDistance < m.retrievalDistance) {
                m.retrievalDistance = c.retrievalDistance;
                m.retrievalSimilarity = c.retrievalSimilarity;
            }
            if (std::find(m.sources.begin(), m.sources.end(), c.source) == m.sources.end()) {
                m.sources.push_back(c.source);
            }
        }
    }

    // --- Filter (TDD 13 steps 4-7): drop reels with an out-of-range id, inactive reels, reels the
    // user has already seen (both via rr::isEligible), and reels with an invalid embedding. Reels
    // are valid by construction (the dataset generator normalizes every embedding), so a cheap
    // non-empty check satisfies step 7 without an O(dim) scan on the hot path.
    std::vector<MergedCandidate> pool;
    pool.reserve(merged.size());
    for (auto &entry : merged) {
        MergedCandidate &m = entry.second;
        if (m.reelId.value >= reels_.size()) {
            continue;
        }
        const Reel &reel = reels_[m.reelId.value];
        if (!isEligible(reel, user) || reel.embedding.empty()) {
            continue;
        }
        pool.push_back(std::move(m));
    }

    // --- Cap the pool at candidateLimit (TDD 13 step 8): keep the best candidateLimit by
    // moreRelevant(). nth_element selects exactly that set; because moreRelevant is a total order,
    // the selection is deterministic. candidatesRanked is the size of the pool actually ranked.
    const std::size_t candidateLimit = static_cast<std::size_t>(request.candidateLimit);
    if (pool.size() > candidateLimit) {
        std::nth_element(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(candidateLimit),
                         pool.end(), moreRelevant);
        pool.resize(candidateLimit);
    }
    response.candidatesRanked = pool.size();

    const std::size_t feedSize = static_cast<std::size_t>(request.feedSize);

    if (ranker_ == nullptr) {
        // --- Ranking stage = IDENTITY (Phase 5, TDD 16.4): order the capped pool by similarity.
        // This sort is the measured ranking latency. Byte-for-byte the pre-Phase-6 behaviour, so
        // every existing test passes unchanged and featureContributions stays empty.
        Stopwatch ranking;
        std::sort(pool.begin(), pool.end(), moreRelevant);
        response.rankingLatencyMs = ranking.elapsedMs();

        // --- Reranking stage = identity (no diversity yet). Measured anyway (~0) so the latency
        // field is populated for every request (diversity lands in Phase 9).
        Stopwatch reranking;
        response.rerankingLatencyMs = reranking.elapsedMs();

        // Phase 22 served-time capture (contracts §7): identity path — the full served pool is
        // `pool` in best-first (retrieval-similarity) order; served score == retrieval similarity.
        // Captured HERE, before the feed loop below moves each MergedCandidate's source vector out.
        // Gate-guarded: request.capture is set (and its extractor non-null) only for a sampled
        // logging request, so a production/golden run does zero extra work (byte-identical, D17).
        if (request.capture != nullptr && request.capture->extractor != nullptr) {
            std::vector<Candidate> capturePool;
            capturePool.reserve(pool.size());
            std::vector<std::vector<CandidateSource>> captureSources;
            captureSources.reserve(pool.size());
            for (const MergedCandidate &m : pool) {
                Candidate c{};
                c.reelId = m.reelId;
                c.source = representativeSource(m.sources); // Exploration wins if present
                c.retrievalDistance = m.retrievalDistance;
                c.retrievalSimilarity = m.retrievalSimilarity;
                c.rankingScore =
                    m.retrievalSimilarity; // identity stage: served score == similarity
                capturePool.push_back(std::move(c));
                captureSources.push_back(m.sources);
            }
            fillRankingCapture(*request.capture, user, request.requestTime, capturePool,
                               captureSources);
        }

        // --- Truncate to the feed size, assign ranks 0..n-1, score = similarity, propagate the
        // label vector into each RankedReel (multi-label per TDD 8.5/8.7). The 4-field aggregate
        // init leaves featureContributions default-empty.
        const std::size_t feedCount = std::min(feedSize, pool.size());
        response.reels.reserve(feedCount);
        for (std::size_t i = 0; i < feedCount; ++i) {
            MergedCandidate &m = pool[i];
            response.reels.push_back(
                RankedReel{m.reelId, m.retrievalSimilarity, i, std::move(m.sources)});
        }

        response.totalLatencyMs = total.elapsedMs();
        return response;
    }

    // --- Ranking stage = the pluggable second-stage Ranker (Phase 6). Inside the ranking
    // Stopwatch: materialize the capped pool as std::vector<Candidate> and hand it to the ranker,
    // which RE-ORDERS (does not change membership). The Candidate carries only the FIRST source
    // label in the first-seen union order; we keep our own reelId -> full-label bookkeeping so the
    // complete multi-source label set still reaches each RankedReel after the (possibly reordered)
    // rank call.
    Stopwatch ranking;
    std::unordered_map<ReelId, std::vector<CandidateSource>> labelsById;
    labelsById.reserve(pool.size());
    std::vector<Candidate> rankPool;
    rankPool.reserve(pool.size());
    for (MergedCandidate &m : pool) {
        Candidate c{};
        c.reelId = m.reelId;
        c.source = representativeSource(m.sources); // Exploration wins the slot if present.
        c.retrievalDistance = m.retrievalDistance;
        c.retrievalSimilarity = m.retrievalSimilarity;
        c.rankingScore = 0.0f;
        rankPool.push_back(std::move(c));
        labelsById.emplace(m.reelId, std::move(m.sources));
    }
    std::vector<Candidate> ranked = ranker_->rank(user, rankPool, request.requestTime);
    response.rankingLatencyMs = ranking.elapsedMs();

    // --- Reranking stage. Guaranteed exploration slots (Phase 8, TDD 12.7 task 3) reorder the
    // ranked pool in place so the feed prefix meets the exploration guarantee (inert unless an
    // exploration source + non-zero guaranteedSlots were supplied and a gate fired). Then, on this
    // RANKED path only, diversity re-ranking (Phase 9, TDD 15) runs when a Reranker was supplied
    // AND request.enableDiversity is set.
    Stopwatch reranking;
    applyExplorationGuarantee(ranked, feedSize);

    // Phase 22 served-time capture (contracts §7): ranked path — `ranked` is now the full served
    // pool best-first (post-exploration-guarantee); each Candidate carries its served rankingScore,
    // retrieval similarity, and representative source, and labelsById still holds the FULL source
    // union (untouched until the truncation loop below). Diversity reranking reorders only the
    // FEED, never this pool order, so poolRank is stable across both ranked sub-paths.
    // Gate-guarded: only a sampled logging request sets request.capture, so production/golden runs
    // do nothing (D17).
    if (request.capture != nullptr && request.capture->extractor != nullptr) {
        std::vector<std::vector<CandidateSource>> captureSources;
        captureSources.reserve(ranked.size());
        for (const Candidate &c : ranked) {
            auto it = labelsById.find(c.reelId);
            captureSources.push_back(
                it != labelsById.end() ? it->second : std::vector<CandidateSource>{c.source});
        }
        fillRankingCapture(*request.capture, user, request.requestTime, ranked, captureSources);
    }

    if (reranker_ != nullptr && request.enableDiversity) {
        // The exploration guarantee already reordered `ranked` (best-first, promotions applied), so
        // the reranker walks THAT order — a constraint selection therefore takes promoted
        // exploration items unless they violate a hard cap (caps win; delivered exploration slots
        // may fall below g; deterministic and accepted). The reranker returns the feed already
        // truncated to feedSize with ranks 0..n-1 and score = each candidate's rankingScore; it
        // sees only the representative single label per reel, so we OVERWRITE each
        // RankedReel.sources with the FULL merged label set from labelsById (falling back to the
        // reranker-provided label if, impossibly, it returned an id we did not send).
        std::vector<RankedReel> reranked = reranker_->rerank(user, ranked, feedSize);
        response.rerankingLatencyMs = reranking.elapsedMs();
        for (RankedReel &r : reranked) {
            if (auto it = labelsById.find(r.reelId); it != labelsById.end()) {
                r.sources = it->second;
            }
        }
        response.reels = std::move(reranked);
        response.totalLatencyMs = total.elapsedMs();
        return response;
    }
    response.rerankingLatencyMs = reranking.elapsedMs();

    // --- Truncate to the feed size in the ranker's returned order; assign ranks 0..n-1, score =
    // the candidate's rankingScore, move featureContributions off the candidate, and restore the
    // FULL merged label set from our bookkeeping (falling back to the candidate's single label if,
    // impossibly, the ranker returned an id we did not send).
    const std::size_t feedCount = std::min(feedSize, ranked.size());
    response.reels.reserve(feedCount);
    for (std::size_t i = 0; i < feedCount; ++i) {
        Candidate &c = ranked[i];
        RankedReel rr{};
        rr.reelId = c.reelId;
        rr.score = c.rankingScore;
        rr.rank = i;
        if (auto it = labelsById.find(c.reelId); it != labelsById.end()) {
            rr.sources = std::move(it->second);
        } else {
            rr.sources = {c.source};
        }
        rr.featureContributions = std::move(c.featureContributions);
        response.reels.push_back(std::move(rr));
    }

    response.totalLatencyMs = total.elapsedMs();
    return response;
}

} // namespace rr
