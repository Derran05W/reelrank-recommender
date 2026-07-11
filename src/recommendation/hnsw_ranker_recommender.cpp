#include "rr/recommendation/hnsw_ranker_recommender.hpp"

#include "rr/domain/candidate.hpp"
#include "rr/infrastructure/config.hpp"

namespace rr {

HNSWRankerRecommender::HNSWRankerRecommender(const RecommenderDeps &deps, Rng rng)
    : reels_(deps.reels), users_(deps.users),
      // D8 draw order: exactly ONE rng draw — the HNSW index seed — identical to HNSWRecommender,
      // so the two algorithms retrieve from byte-identical graphs under the same master seed and
      // any feed difference is attributable to the ranking stage alone.
      index_(deps.config.simulation.dimensions, deps.config.hnsw, rng.nextU64()),
      hnswSource_(index_), popularSource_(reels_, deps.config.recommendation.popularCandidates),
      trendingSource_(reels_, deps.config.recommendation.trendingCandidates,
                      deps.config.ranking.trendingHalfLifeSeconds),
      creatorSource_(reels_, deps.config.recommendation.creatorAffinityCandidates),
      ranker_(reels_, deps.config.ranking),
      orchestrator_({&hnswSource_, &popularSource_, &trendingSource_, &creatorSource_}, reels_,
                    &ranker_) {
    // Build the graph once over all active reels; embeddings are immutable (D2). Mirrors
    // HNSWRecommender exactly.
    for (const Reel &reel : reels_) {
        if (reel.active) {
            index_.insert(reel.id, reel.embedding);
        }
    }
}

RecommendationResponse HNSWRankerRecommender::recommend(const RecommendationRequest &request) {
    const User &user = users_[request.userId.value];
    return orchestrator_.recommend(user, request);
}

std::string HNSWRankerRecommender::name() const {
    return toString(RecommendationAlgorithm::HnswRanker);
}

const VectorIndex *HNSWRankerRecommender::retrievalIndex() const { return &index_; }

} // namespace rr
