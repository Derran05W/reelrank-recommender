#include "rr/recommendation/full_recommender.hpp"

#include <utility>

#include "rr/domain/candidate.hpp"
#include "rr/infrastructure/config.hpp"

namespace rr {

namespace {

// Source registration order fixes the first-seen label union order in the merge (TDD 13 listing:
// HNSW, popular, trending, fresh, creator affinity, exploration); the exploration source is last
// and the Orchestrator elects Exploration as the representative label whenever present. In
// isolation mode the list is EXACTLY hnsw_ranker's four sources.
std::vector<CandidateGenerator *>
makeSourceList(HNSWCandidateSource &hnsw, PopularCandidateSource &popular,
               TrendingCandidateSource &trending, FreshCandidateSource &fresh,
               CreatorAffinityCandidateSource &creator, ExplorationCandidateSource &exploration,
               bool explorationEnabled) {
    if (!explorationEnabled) {
        return {&hnsw, &popular, &trending, &creator};
    }
    return {&hnsw, &popular, &trending, &fresh, &creator, &exploration};
}

} // namespace

FullRecommender::FullRecommender(const RecommenderDeps &deps, Rng rng)
    : reels_(deps.reels), users_(deps.users), explorationEnabled_(deps.config.exploration.enabled),
      // D8: rng_ is moved from the forked "recommender" stream BEFORE index_ is initialized, so
      // index_'s seed (rng_.nextU64()) is the FIRST draw of that stream — byte-identical to hnsw,
      // hnsw_ranker, and hnsw_ranker_exploration. Epsilon gate draws (complete mode) come later.
      rng_(std::move(rng)),
      index_(deps.config.simulation.dimensions, deps.config.hnsw, rng_.nextU64()),
      hnswSource_(index_), popularSource_(reels_, deps.config.recommendation.popularCandidates),
      trendingSource_(reels_, deps.config.recommendation.trendingCandidates,
                      deps.config.ranking.trendingHalfLifeSeconds),
      creatorSource_(reels_, deps.config.recommendation.creatorAffinityCandidates),
      freshSource_(reels_, deps.config.recommendation.freshCandidates,
                   deps.config.exploration.freshWindowSeconds),
      explorationSource_(reels_, deps.config.exploration.epsilon,
                         deps.config.recommendation.explorationCandidates,
                         deps.config.exploration.freshWindowSeconds, &rng_),
      ranker_(reels_, deps.config.ranking), reranker_(reels_, deps.config.diversity),
      orchestrator_(makeSourceList(hnswSource_, popularSource_, trendingSource_, freshSource_,
                                   creatorSource_, explorationSource_, explorationEnabled_),
                    reels_, &ranker_, explorationEnabled_ ? &explorationSource_ : nullptr,
                    explorationEnabled_ ? deps.config.exploration.guaranteedSlots : 0, &reranker_) {
    // Build the graph once over all active reels; embeddings are immutable (D2). Mirrors the
    // other HNSW recommenders exactly.
    for (const Reel &reel : reels_) {
        if (reel.active) {
            index_.insert(reel.id, reel.embedding);
        }
    }
}

RecommendationResponse FullRecommender::recommend(const RecommendationRequest &request) {
    const User &user = users_[request.userId.value];
    return orchestrator_.recommend(user, request);
}

std::string FullRecommender::name() const {
    return toString(RecommendationAlgorithm::HnswRankerDiversity);
}

const VectorIndex *FullRecommender::retrievalIndex() const { return &index_; }

void FullRecommender::onReelsAppended(size_t firstNewIndex) {
    // Same eligibility rule as the constructor: appended reels are indexed once, insert-only (D2).
    for (size_t i = firstNewIndex; i < reels_.size(); ++i) {
        const Reel &reel = reels_[i];
        if (reel.active) {
            index_.insert(reel.id, reel.embedding);
        }
    }
}

} // namespace rr
