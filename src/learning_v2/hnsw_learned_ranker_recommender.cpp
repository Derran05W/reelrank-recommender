#include "rr/learning_v2/hnsw_learned_ranker_recommender.hpp"

#include "rr/domain/candidate.hpp"
#include "rr/infrastructure/config.hpp"

namespace rr {

HNSWLearnedRankerRecommender::HNSWLearnedRankerRecommender(const RecommenderDeps &deps, Rng rng)
    : reels_(deps.reels), users_(deps.users),
      // D8 draw order: exactly ONE rng draw — the HNSW index seed — identical to HNSWRecommender /
      // HNSWRankerRecommender, so the learned arm retrieves from a byte-identical graph under the
      // same master seed and any feed difference is attributable to the ranking stage alone.
      index_(deps.config.simulation.dimensions, deps.config.hnsw, rng.nextU64()),
      hnswSource_(index_), popularSource_(reels_, deps.config.recommendation.popularCandidates),
      trendingSource_(reels_, deps.config.recommendation.trendingCandidates,
                      deps.config.ranking.trendingHalfLifeSeconds),
      creatorSource_(reels_, deps.config.recommendation.creatorAffinityCandidates),
      ranker_(reels_, deps.config.ranking, deps.config.realism.contentV2,
              deps.config.realism.personalizedDiversity, deps.config.learningV2.valueWeights),
      orchestrator_({&hnswSource_, &popularSource_, &trendingSource_, &creatorSource_}, reels_,
                    &ranker_) {
    // Build the graph once over all active reels; embeddings are immutable (D2). Mirrors
    // HNSWRankerRecommender exactly.
    for (const Reel &reel : reels_) {
        if (reel.active) {
            index_.insert(reel.id, reel.embedding);
        }
    }
}

RecommendationResponse
HNSWLearnedRankerRecommender::recommend(const RecommendationRequest &request) {
    const User &user = users_[request.userId.value];
    return orchestrator_.recommend(user, request);
}

std::string HNSWLearnedRankerRecommender::name() const {
    return toString(RecommendationAlgorithm::HnswLearnedRanker);
}

const VectorIndex *HNSWLearnedRankerRecommender::retrievalIndex() const { return &index_; }

void HNSWLearnedRankerRecommender::onReelsAppended(size_t firstNewIndex) {
    for (size_t i = firstNewIndex; i < reels_.size(); ++i) {
        const Reel &reel = reels_[i];
        if (reel.active) {
            index_.insert(reel.id, reel.embedding);
        }
    }
}

} // namespace rr
