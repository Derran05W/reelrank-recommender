#pragma once

#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

namespace rr {

// TDD 12.1: the HNSW vector candidate generator. Queries a recommender-owned HNSWVectorIndex
// (held by non-owning reference) with the user's effective preference for the top
// request.candidateLimit reels and emits one Candidate{source=VectorHNSW, ...} per result.
//
// Similarity is recovered from Euclidean distance via rr::similarityFromEuclidean: for unit
// vectors similarity = 1 - d^2/2 = cos (design decision D3, which deliberately OVERRIDES the
// TDD 12.1 1/(1+d) suggestion so exact-vs-ANN comparisons stay metric-identical).
//
// This source returns RAW index results with NO active/seen/validity filtering — merge and
// filtering are the Orchestrator's job (TDD 13 ordering).
class HNSWCandidateSource final : public CandidateGenerator {
  public:
    explicit HNSWCandidateSource(const HNSWVectorIndex &index);

    std::vector<Candidate> generate(const User &user,
                                    const RecommendationRequest &request) override;

  private:
    const HNSWVectorIndex &index_;
};

} // namespace rr
