#include "rr/recommendation/orchestrator.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"
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

} // namespace

Orchestrator::Orchestrator(std::vector<CandidateGenerator *> sources,
                           const std::vector<Reel> &reels)
    : sources_(std::move(sources)), reels_(reels) {}

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

    // --- Ranking stage = identity for Phase 5 (TDD 16.4): order the capped pool by similarity.
    // This sort is the measured ranking latency.
    Stopwatch ranking;
    std::sort(pool.begin(), pool.end(), moreRelevant);
    response.rankingLatencyMs = ranking.elapsedMs();

    // --- Reranking stage = identity in Phase 5 (no diversity yet). Measured anyway (~0) so the
    // latency field is populated for every request (diversity lands in Phase 9).
    Stopwatch reranking;
    response.rerankingLatencyMs = reranking.elapsedMs();

    // --- Truncate to the feed size, assign ranks 0..n-1, score = similarity, propagate the label
    // vector into each RankedReel (multi-label per TDD 8.5/8.7).
    const std::size_t feedSize = static_cast<std::size_t>(request.feedSize);
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

} // namespace rr
