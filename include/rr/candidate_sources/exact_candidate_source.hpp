#pragma once

#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"
#include "rr/vindex/exact_vector_index.hpp"

namespace rr {

// TDD 12.2: the exact vector candidate generator — the ground-truth / quality-ceiling mirror of
// HNSWCandidateSource. Queries a recommender-owned ExactVectorIndex (brute-force, held by
// non-owning reference) with the user's effective preference for the top request.candidateLimit
// reels and emits one Candidate{source=VectorExact, ...} per result.
//
// Similarity is recovered from Euclidean distance via rr::similarityFromEuclidean (D3; overrides
// the TDD 12.1 1/(1+d) suggestion). Like every source it returns RAW results — the Orchestrator
// performs merge and filtering (TDD 13).
class ExactCandidateSource final : public CandidateGenerator {
  public:
    explicit ExactCandidateSource(const ExactVectorIndex &index);

    std::vector<Candidate> generate(const User &user,
                                    const RecommendationRequest &request) override;

  private:
    const ExactVectorIndex &index_;
};

} // namespace rr
