#pragma once

#include <vector>

#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"

namespace rr {

// TDD 7 / 13: the recommendation orchestrator. Given a fixed set of candidate sources it runs
// them, merges + deduplicates their output (preserving all source labels), drops ineligible
// reels, caps the pool, applies an (identity, in Phase 5) ranking stage, and truncates to the
// requested feed size — recording per-stage wall-clock latency into the response.
//
// The whole path is DETERMINISTIC and exception-free: no rr::Rng is used, and every ordering
// decision falls back to ascending ReelId so the output is independent of hash-map iteration
// order.
//
// Ownership: the Orchestrator holds NON-OWNING pointers to its sources and a reference to the
// live reels vector. The owner (a Recommender) must keep the sources and reels alive for the
// Orchestrator's lifetime.
class Orchestrator {
  public:
    Orchestrator(std::vector<CandidateGenerator *> sources, const std::vector<Reel> &reels);

    RecommendationResponse recommend(const User &user, const RecommendationRequest &request);

  private:
    std::vector<CandidateGenerator *> sources_;
    const std::vector<Reel> &reels_;
};

} // namespace rr
