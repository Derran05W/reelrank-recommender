#pragma once

#include <string>

#include "rr/domain/recommendation.hpp"

namespace rr {

// TDD 16: top-level recommender. Concrete algorithms (random, popularity, HNSW, ...) arrive in
// later phases.
class Recommender {
  public:
    virtual RecommendationResponse recommend(const RecommendationRequest &request) = 0;

    virtual std::string name() const = 0;

    virtual ~Recommender() = default;
};

} // namespace rr
