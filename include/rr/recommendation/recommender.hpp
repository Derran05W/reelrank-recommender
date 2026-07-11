#pragma once

#include <string>

#include "rr/domain/recommendation.hpp"

namespace rr {

class VectorIndex;

// TDD 16: top-level recommender. Concrete algorithms (random, popularity, HNSW, ...) arrive in
// later phases.
class Recommender {
  public:
    virtual RecommendationResponse recommend(const RecommendationRequest &request) = 0;

    virtual std::string name() const = 0;

    // The retrieval index this recommender queries, if it is vector-based; nullptr otherwise
    // (default). Evaluation-only hook (TDD 18.1): the harness measures live Recall@K / distance
    // error by querying this index against an exact ground-truth index on sampled requests. Never
    // used on the recommendation path itself.
    virtual const VectorIndex *retrievalIndex() const { return nullptr; }

    virtual ~Recommender() = default;
};

} // namespace rr
