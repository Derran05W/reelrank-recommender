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
#include "rr/learning_v2/learned_ranker.hpp"
#include "rr/recommendation/orchestrator.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

namespace rr {

// Phase 23 (contracts §1/§7): the `hnsw_learned_ranker` model. Byte-for-byte the
// HNSWRankerRecommender pipeline — the same four Phase-6 candidate sources (HNSW / popular /
// trending / creator-affinity) merged + deduped by the Orchestrator, the same single
// "recommender"-stream rng draw for the HNSW index seed, diversity off by default — with the
// WeightedRanker REPLACED by a LearnedRanker serving the §4.21 multi-objective value. The
// LearnedRanker owns its own WeightedRanker for the cold-start fallback, so a modelless run ranks
// exactly like hnsw_ranker (byte-identical fallback scores).
//
// Lives in rr_learning_v2 (which depends on rr_recommend) because it owns a LearnedRanker; the
// recommendation-side factory therefore cannot construct it without a library cycle. The
// EventDrivenRunner constructs it directly (it also needs the retrainer wiring + a live handle to
// the ranker for deterministic model hot-swaps) — the same event-only-construction pattern as the
// oracle arm. `learnedRanker()` exposes the ranker so the runner can setModels() between requests
// and read the fallback-share counters at result assembly.
class HNSWLearnedRankerRecommender final : public Recommender {
  public:
    HNSWLearnedRankerRecommender(const RecommenderDeps &deps, Rng rng);

    RecommendationResponse recommend(const RecommendationRequest &request) override;

    std::string name() const override;

    const VectorIndex *retrievalIndex() const override;

    void onReelsAppended(size_t firstNewIndex) override;

    // Non-owning access to the served ranker for the runner's retrain hook (model hot-swap) and
    // fallback accounting. Valid for the recommender's lifetime.
    learning_v2::LearnedRanker &learnedRanker() { return ranker_; }
    const learning_v2::LearnedRanker &learnedRanker() const { return ranker_; }

  private:
    const std::vector<Reel> &reels_;
    const std::vector<User> &users_;
    HNSWVectorIndex index_;
    HNSWCandidateSource hnswSource_;
    PopularCandidateSource popularSource_;
    TrendingCandidateSource trendingSource_;
    CreatorAffinityCandidateSource creatorSource_;
    learning_v2::LearnedRanker ranker_;
    Orchestrator orchestrator_;
};

} // namespace rr
