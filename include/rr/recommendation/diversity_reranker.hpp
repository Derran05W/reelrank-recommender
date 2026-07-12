#pragma once

#include <cstddef>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/recommendation/constraint_reranker.hpp"
#include "rr/recommendation/mmr_reranker.hpp"
#include "rr/recommendation/reranker.hpp"

namespace rr {

// Composite diversity re-ranker (Phase 9, plan task 2's composition). Facade over the two
// primitives that the phase-9 experiment arms compare.
//
// COMPOSITION DECISION (the load-bearing design choice for this phase).
// Constraints are HARD rules and select the feed SET; MMR is an ORDERING heuristic and orders
// WITHIN that set. So:
//   1. ConstraintReranker.selectFeed picks the feed SET by the TDD 15.1 hard rules, walking the
//      input in relevance order (respecting the Orchestrator's positional exploration promotions).
//   2a. If config.useMmr: MMRReranker orders that set (feedSize = set size, relevance min-max
//       normalised WITHIN the selected set). This MMR ordering REPLACES the ConstraintReranker's
//       consecutive-same-topic swap pass: both are ordering heuristics over an already-chosen set,
//       and MMR's diversity objective subsumes the swap's "avoid consecutive same-topic" intent, so
//       running both would be redundant. Hence when useMmr we call selectFeed (which omits the
//       swap) and let MMR do all ordering.
//   2b. If !config.useMmr: the output is exactly ConstraintReranker.rerank — the selected set in
//       relevance order with the consecutive-topic swap applied — byte-for-byte, no MMR involved.
// Either way, ranks are reassigned 0..n-1 over the final order (both components already do this).
//
// The constraint stage owns EVERY hard invariant (no dup/seen ids, creator/topic caps): MMR only
// permutes the set it is handed, so a DiversityReranker feed can never violate a cap regardless of
// useMmr. The flagship property suite asserts exactly this across both modes.
//
// Implemented by owning both components (each independently unit-tested); non-owning reference to
// the reels vector flows through to both. Deterministic and exception-free like its parts.
class DiversityReranker final : public Reranker {
  public:
    DiversityReranker(const std::vector<Reel> &reels, const DiversityConfig &config);

    std::vector<RankedReel> rerank(const User &user, const std::vector<Candidate> &rankedCandidates,
                                   std::size_t feedSize) const override;

  private:
    DiversityConfig config_;
    ConstraintReranker constraint_;
    MMRReranker mmr_;
};

} // namespace rr
