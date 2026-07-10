#pragma once

#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/user.hpp"

namespace rr {

// TDD 12: produces candidate reels for a user given a request.
class CandidateGenerator {
  public:
    virtual std::vector<Candidate> generate(const User &user,
                                            const RecommendationRequest &request) = 0;

    virtual ~CandidateGenerator() = default;
};

} // namespace rr
