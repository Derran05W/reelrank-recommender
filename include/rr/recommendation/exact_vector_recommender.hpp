#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/vindex/exact_vector_index.hpp"

namespace rr {

// TDD 16.3 - the personalization ceiling. Brute-force exact nearest-neighbour retrieval over the
// user's effective preference. The index is built once over all active reels in the constructor
// (embeddings are immutable, D2) and reused for every request. Ascending Euclidean distance over
// unit vectors is exactly descending cosine similarity (D3).
class ExactVectorRecommender final : public Recommender {
  public:
    ExactVectorRecommender(const RecommenderDeps &deps, Rng rng);

    RecommendationResponse recommend(const RecommendationRequest &request) override;

    std::string name() const override;

    // Evaluation hook (TDD 18.1): the exact index this recommender queries IS its own ground truth,
    // so live exact-vs-exact recall through the harness measures exactly 1.0 recall / 0.0 distance
    // error — a wiring self-check. Never used on the recommendation path itself.
    const VectorIndex *retrievalIndex() const override;

  private:
    // Walk `results` in ascending-distance order, emit the eligible ones (active + unseen) as
    // ranked reels until feedSize is reached, assigning rank 0..n-1 and score = cosine similarity.
    void appendEligible(const std::vector<VectorSearchResult> &results, const User &user,
                        std::size_t feedSize, RecommendationResponse &response) const;

    const std::vector<Reel> &reels_;
    const std::vector<User> &users_;
    ExactVectorIndex index_;
};

} // namespace rr
