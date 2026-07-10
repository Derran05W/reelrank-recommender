#pragma once

#include <cstddef>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/user.hpp"

namespace rr {

// TDD 23.3.
class Reranker {
  public:
    virtual std::vector<RankedReel> rerank(const User &user,
                                           const std::vector<Candidate> &rankedCandidates,
                                           size_t feedSize) const = 0;

    virtual ~Reranker() = default;
};

} // namespace rr
