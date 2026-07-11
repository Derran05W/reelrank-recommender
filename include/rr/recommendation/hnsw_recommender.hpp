#pragma once

#include <string>
#include <vector>

#include "rr/candidate_sources/hnsw_candidate_source.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/orchestrator.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

namespace rr {

// TDD 16.4: the HNSW-only recommender — ANN retrieval with similarity ordering and no learned
// ranking. It builds one HNSWVectorIndex over every ACTIVE reel at construction (embeddings are
// immutable, D2) and serves each request through an Orchestrator wired with a single
// HNSWCandidateSource. Its feed quality is pure ANN recall measured against the exact ceiling
// (ExactVectorRecommender).
//
// Member order is load-bearing: index_ is built first (its seed is the single rng draw), then the
// source referencing it, then the orchestrator referencing the source and the reels vector.
class HNSWRecommender final : public Recommender {
  public:
    HNSWRecommender(const RecommenderDeps &deps, Rng rng);

    RecommendationResponse recommend(const RecommendationRequest &request) override;

    std::string name() const override;

    // Evaluation-only hook (TDD 18.1): expose the HNSW graph so the harness can measure live
    // Recall@K / distance error against exact ground truth. Never touched on the feed path.
    const VectorIndex *retrievalIndex() const override;

  private:
    const std::vector<Reel> &reels_;
    const std::vector<User> &users_;
    HNSWVectorIndex index_;
    HNSWCandidateSource source_;
    Orchestrator orchestrator_;
};

} // namespace rr
