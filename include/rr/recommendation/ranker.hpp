#pragma once

#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// TDD 23.2.
class Ranker {
  public:
    virtual std::vector<Candidate> rank(const User &user, const std::vector<Candidate> &candidates,
                                        Timestamp now) const = 0;

    virtual ~Ranker() = default;
};

} // namespace rr
