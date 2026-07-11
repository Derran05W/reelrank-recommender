#pragma once

#include <vector>

#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"
#include "rr/recommendation/ranker.hpp"

namespace rr {

// TDD 7 / 13: the recommendation orchestrator. Given a fixed set of candidate sources it runs
// them, merges + deduplicates their output (preserving all source labels), drops ineligible
// reels, caps the pool, applies a ranking stage, and truncates to the requested feed size —
// recording per-stage wall-clock latency into the response.
//
// The ranking stage is pluggable (Phase 6). If `ranker == nullptr` the stage is IDENTITY: the
// capped pool is ordered by descending retrieval similarity (Phase 5 behaviour, byte-for-byte
// unchanged). If a Ranker is supplied, the capped pool is materialized as std::vector<Candidate>
// and handed to `ranker->rank(...)`, which RE-ORDERS it (it never changes membership); each
// resulting RankedReel then carries the ranker's rankingScore and featureContributions while the
// FULL merged multi-source label set is still preserved.
//
// The whole path is DETERMINISTIC and exception-free: no rr::Rng is used, and every ordering
// decision falls back to ascending ReelId so the output is independent of hash-map iteration
// order.
//
// Ownership: the Orchestrator holds NON-OWNING pointers to its sources and the ranker and a
// reference to the live reels vector. The owner (a Recommender) must keep the sources, ranker,
// and reels alive for the Orchestrator's lifetime.
class Orchestrator {
  public:
    Orchestrator(std::vector<CandidateGenerator *> sources, const std::vector<Reel> &reels,
                 const Ranker *ranker = nullptr);

    RecommendationResponse recommend(const User &user, const RecommendationRequest &request);

  private:
    std::vector<CandidateGenerator *> sources_;
    const std::vector<Reel> &reels_;
    const Ranker *ranker_;
};

} // namespace rr
