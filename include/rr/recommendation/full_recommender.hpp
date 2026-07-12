#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rr/candidate_sources/creator_affinity_candidate_source.hpp"
#include "rr/candidate_sources/exploration_candidate_source.hpp"
#include "rr/candidate_sources/fresh_candidate_source.hpp"
#include "rr/candidate_sources/hnsw_candidate_source.hpp"
#include "rr/candidate_sources/popular_candidate_source.hpp"
#include "rr/candidate_sources/trending_candidate_source.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/diversity_reranker.hpp"
#include "rr/recommendation/orchestrator.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/recommendation/weighted_ranker.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

namespace rr {

// TDD 16.6: HNSW retrieval + the WeightedRanker + diversity re-ranking — and, when exploration is
// enabled, the COMPLETE initial system (phase-9 plan task 4). Registered as
// "hnsw_ranker_diversity".
//
// TWO MODES, selected by config.exploration.enabled at construction (recorded for commit.md):
//  - Diversity-isolation mode (exploration.enabled = false): sources are EXACTLY hnsw_ranker's
//    four (HNSW, popular, trending, creator-affinity) plus the DiversityReranker. The phase-9
//    experiment arms (hnsw_ranker vs +constraints vs +constraints+MMR) run in this mode so the
//    re-ranker is the ONLY variable; with request.enableDiversity false the output is
//    byte-identical to hnsw_ranker (the no-op regression check, mirroring Phase 8's epsilon=0).
//  - Complete-initial-system mode (exploration.enabled = true): adds the UNGATED
//    FreshCandidateSource at recommendation.freshCandidates (TDD 13's fresh:100 merge count,
//    deliberately deferred from Phase 8 to here — see the Phase 8 commit.md deviation) and the
//    epsilon-gated ExplorationCandidateSource with guaranteed slots, exactly as
//    HNSWExplorationRecommender wires them. This is "the complete initial system": every TDD 13
//    source, ranking, exploration protection, and diversity re-ranking in one pipeline.
// The fresh source runs WITHOUT the topic-proximity filter: relevance shaping is the ranker's job
// (its similarity feature already scores fresh candidates); the source stays a pure recency
// window per TDD 12.5's base definition.
//
// Both sources are constructed unconditionally (construction is cheap and draws no rng) but
// registered with the Orchestrator only in complete mode, so in isolation mode they never
// generate and never draw — keeping the recommender rng stream byte-identical to hnsw_ranker's.
//
// D8 CONTRACT (identical to the other HNSW recommenders): the recommender OWNS its forked
// "recommender" rng; the HNSW index seed is that rng's FIRST draw, so hnsw / hnsw_ranker /
// hnsw_ranker_exploration / hnsw_ranker_diversity all build byte-identical graphs from one master
// seed. Per-request epsilon gate draws (complete mode only) come after, from the same rng.
// Member order is load-bearing: rng_ before index_ (seed = first draw) and before
// explorationSource_ (which holds a non-owning pointer back to rng_).
//
// DIVERSITY GATE: the Orchestrator applies reranker_ only when request.enableDiversity is set
// (config-driven: the harness copies config.diversity.enabled). The caps themselves and the
// constraints/MMR composition live in DiversityReranker; the interplay with the exploration
// guarantee (caps take precedence) is documented at the Orchestrator's rerank call site.
class FullRecommender final : public Recommender {
  public:
    FullRecommender(const RecommenderDeps &deps, Rng rng);

    RecommendationResponse recommend(const RecommendationRequest &request) override;

    std::string name() const override;

    // Evaluation-only hook (TDD 18.1), as the other HNSW recommenders: live Recall@K / distance
    // error stay measurable. Never touched on the feed path.
    const VectorIndex *retrievalIndex() const override;

    // Inserts appended ACTIVE reels into the HNSW graph (D2 insert-only).
    void onReelsAppended(size_t firstNewIndex) override;

  private:
    const std::vector<Reel> &reels_;
    const std::vector<User> &users_;
    bool explorationEnabled_;
    Rng rng_; // owned; FIRST draw = index seed (D8), later draws = epsilon gates (complete mode)
    HNSWVectorIndex index_;
    HNSWCandidateSource hnswSource_;
    PopularCandidateSource popularSource_;
    TrendingCandidateSource trendingSource_;
    CreatorAffinityCandidateSource creatorSource_;
    FreshCandidateSource freshSource_;
    ExplorationCandidateSource explorationSource_;
    WeightedRanker ranker_;
    DiversityReranker reranker_;
    Orchestrator orchestrator_;
};

} // namespace rr
