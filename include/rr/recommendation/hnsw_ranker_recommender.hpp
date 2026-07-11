#pragma once

#include <string>
#include <vector>

#include "rr/candidate_sources/creator_affinity_candidate_source.hpp"
#include "rr/candidate_sources/hnsw_candidate_source.hpp"
#include "rr/candidate_sources/popular_candidate_source.hpp"
#include "rr/candidate_sources/trending_candidate_source.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/orchestrator.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/recommendation/weighted_ranker.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

namespace rr {

// TDD 16.5: the main production-style model — HNSW retrieval plus the full second-stage pipeline:
// all four Phase-6 candidate sources (HNSW via request.candidateLimit = vector_candidates,
// popular/trending/creator-affinity at their configured counts, TDD 13) merged and deduped by the
// Orchestrator, then ordered by the WeightedRanker's feature score instead of raw similarity.
//
// Member order is load-bearing (mirrors HNSWRecommender): index_ first — its seed is the single
// rng draw of the forked "recommender" stream (D8), so hnsw and hnsw_ranker build byte-identical
// graphs from the same seed — then the sources referencing it, then the ranker, then the
// orchestrator referencing all of them. Source order fixes the first-seen label union order in
// the merge (HNSW first, per the TDD 13 listing).
class HNSWRankerRecommender final : public Recommender {
  public:
    HNSWRankerRecommender(const RecommenderDeps &deps, Rng rng);

    RecommendationResponse recommend(const RecommendationRequest &request) override;

    std::string name() const override;

    // Evaluation-only hook (TDD 18.1), identical to HNSWRecommender's: live Recall@K / distance
    // error stay measurable when the ranker re-orders feeds. Never touched on the feed path.
    const VectorIndex *retrievalIndex() const override;

  private:
    const std::vector<Reel> &reels_;
    const std::vector<User> &users_;
    HNSWVectorIndex index_;
    HNSWCandidateSource hnswSource_;
    PopularCandidateSource popularSource_;
    TrendingCandidateSource trendingSource_;
    CreatorAffinityCandidateSource creatorSource_;
    WeightedRanker ranker_;
    Orchestrator orchestrator_;
};

} // namespace rr
