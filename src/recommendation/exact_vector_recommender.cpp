#include "rr/recommendation/exact_vector_recommender.hpp"

#include <algorithm>
#include <cstddef>

#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/recommendation/seen_filter.hpp"

namespace rr {

ExactVectorRecommender::ExactVectorRecommender(const RecommenderDeps &deps, Rng /*rng*/)
    : reels_(deps.reels), users_(deps.users), index_(deps.config.simulation.dimensions) {
    // Build the exact index once over all active reels; embeddings are immutable (D2). Insert
    // validates dimension/finiteness and throws std::invalid_argument on a bad embedding, which
    // is a setup error (D10) and correctly surfaces here in the constructor.
    for (const Reel &reel : reels_) {
        if (reel.active) {
            index_.insert(reel.id, reel.embedding);
        }
    }
}

void ExactVectorRecommender::appendEligible(const std::vector<VectorSearchResult> &results,
                                            const User &user, std::size_t feedSize,
                                            RecommendationResponse &response) const {
    for (const VectorSearchResult &result : results) {
        if (response.reels.size() >= feedSize) {
            break;
        }
        // Dense ids: the live reel for a result is reels_[id.value]. Re-check eligibility against
        // live state (a reel indexed while active may have been deactivated since; seen reels are
        // dropped here per TDD 13 item 5).
        const Reel &reel = reels_[result.reelId.value];
        if (!isEligible(reel, user)) {
            continue;
        }
        response.reels.push_back(RankedReel{result.reelId,
                                            similarityFromEuclidean(result.distance),
                                            response.reels.size(),
                                            {CandidateSource::VectorExact}});
    }
}

RecommendationResponse ExactVectorRecommender::recommend(const RecommendationRequest &request) {
    // Per-stage wall-clock timing (D9: wall clock only for latency). This adds no rng draws and
    // does not change the feed, so determinism is preserved. Latency-field SEMANTICS:
    //   retrieval  = the exact index search(es); ranking = the eligible-walk / feed assembly;
    //   reranking  = 0.0 (this baseline has no reranking stage); total = the whole call.
    //   candidatesRetrieved = search results returned; candidatesRanked = feed items emitted.
    const Stopwatch total;
    double retrievalMs = 0.0;
    double rankingMs = 0.0;

    const User &user = users_[request.userId.value];
    const Embedding &query = effectivePreference(user);

    RecommendationResponse response{};
    response.rerankingLatencyMs = 0.0; // no reranking stage (documented above)
    const std::size_t indexSize = index_.size();
    const std::size_t feedSize = static_cast<std::size_t>(request.feedSize);
    if (indexSize == 0 || feedSize == 0) {
        response.totalLatencyMs = total.elapsedMs();
        return response;
    }

    // Over-fetch so that after dropping the user's seen reels we still have feedSize left.
    std::size_t k = feedSize + user.seenReels.size();
    if (k > indexSize) {
        k = indexSize;
    }
    Stopwatch retrieval;
    std::vector<VectorSearchResult> results = index_.search(query, k);
    retrievalMs += retrieval.elapsedMs();
    response.candidatesRetrieved = results.size();

    Stopwatch ranking;
    appendEligible(results, user, feedSize, response);
    rankingMs += ranking.elapsedMs();

    // Pathological shortfall (e.g. inactive reels or seen reels crowded out the top-k window):
    // fall back to a full-index scan and re-select.
    if (response.reels.size() < feedSize && k < indexSize) {
        Stopwatch retrievalFallback;
        results = index_.search(query, indexSize);
        retrievalMs += retrievalFallback.elapsedMs();
        response.candidatesRetrieved = results.size();
        response.reels.clear();
        Stopwatch rankingFallback;
        appendEligible(results, user, feedSize, response);
        rankingMs += rankingFallback.elapsedMs();
    }

    response.candidatesRanked = response.reels.size();
    response.retrievalLatencyMs = retrievalMs;
    response.rankingLatencyMs = rankingMs;
    response.totalLatencyMs = total.elapsedMs();
    return response;
}

std::string ExactVectorRecommender::name() const { return "exact_vector"; }

const VectorIndex *ExactVectorRecommender::retrievalIndex() const { return &index_; }

} // namespace rr
