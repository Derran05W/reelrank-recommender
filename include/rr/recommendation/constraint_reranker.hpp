#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/recommendation/reranker.hpp"

namespace rr {

// TDD 15.1 constraint-based diversity re-ranker (Phase 9, plan task 1).
//
// SELECTION IS A GREEDY WALK IN THE GIVEN ORDER. rankedCandidates arrives best-first (position ==
// relevance); we walk it once, front to back, and NEVER re-sort it. This is load-bearing: the
// Orchestrator's guaranteed-exploration-slot promotions are POSITIONAL (they move items to the top
// of the pool BEFORE reranking), so re-sorting here would silently undo them. Selection admits a
// candidate iff it violates none of the HARD rules below; the relative order of admitted items is
// therefore exactly their relative order in the input.
//
// HARD RULES (TDD 15.1):
//   * No duplicate reel ids in the feed.
//   * No reel already in user.seenReels. (Belt-and-braces: the Orchestrator filters seen reels
//     upstream, so a standalone/mis-wired caller stays correct and the rule is self-contained.)
//   * At most config.maxPerCreator reels per creator (absolute; TDD 15.1 "maximum two per creator
//     in one feed").
//   * At most topicCap reels per primary topic, where the TDD's "three per ten-item feed" is scaled
//     proportionally to the requested feedSize: topicCap = max(1, ceil(maxPerTopic * feedSize/10)).
//     See topicCap() for the exact rule.
// A candidate whose reelId is out of range for `reels` is treated as ineligible and skipped (same
// belt-and-braces posture; the Orchestrator already drops such ids).
//
// SHORTER-THAN-REQUESTED FEED (plan task 5, documented behaviour). Caps are HARD: there is no
// relax/backfill pass. Selection stops at feedSize admitted items OR when the walk reaches the pool
// end with no further addable candidate. The feed is therefore shorter than feedSize ONLY when no
// remaining candidate can be added without violating a cap (this subsumes literal pool exhaustion).
// Rationale: a diversity floor that is silently traded away for feed length is not a floor;
// delivering a shorter, fully-compliant feed is the honest behaviour and keeps every hard invariant
// property-checkable.
//
// CONSECUTIVE-SAME-TOPIC AVOIDANCE (TDD 15.1 "avoid consecutive same-topic where possible"). After
// selection, a SINGLE greedy forward pass runs over the chosen feed: whenever feed[i] and feed[i+1]
// share a primary topic, feed[i+1] is swapped with the NEAREST later item whose topic differs (if
// any exists). "Nearest" is the unique smallest later index, so the choice is deterministic; the
// pass visits each position once and performs at most one swap per position, so it provably
// terminates. Where a differing later item does not exist the run is left intact ("where
// possible"). This pass is purely cosmetic — it never changes the feed SET, only its order. Any
// residual tie (none arises from unique indices; the input is already a total order) falls back to
// ascending ReelId.
//
// OUTPUT. Each admitted candidate becomes a RankedReel: reelId, score = candidate.rankingScore,
// rank = final 0..n-1 position (after the swap pass), sources = {candidate.source} (the single
// representative label the Orchestrator handed us — the Orchestrator restores the full merged
// multi-source label set afterwards), featureContributions copied verbatim from the candidate.
//
// Fully deterministic and exception-free (no rr::Rng, no wall clock): every ordering decision is a
// walk in the input's total order or falls back to ascending ReelId.
//
// Ownership: NON-OWNING reference to the live reels vector (dense-id lookup reels[id.value]); the
// owner must keep it alive for this reranker's lifetime.
class ConstraintReranker final : public Reranker {
  public:
    ConstraintReranker(const std::vector<Reel> &reels, const DiversityConfig &config);

    // Reranker interface: hard-constraint selection + the consecutive-topic swap pass, emitted as
    // RankedReels (see class doc).
    std::vector<RankedReel> rerank(const User &user, const std::vector<Candidate> &rankedCandidates,
                                   std::size_t feedSize) const override;

    // The hard-constraint feed SET only: the greedy selection above, returning copies of the
    // admitted candidates in relevance-walk order, WITHOUT the cosmetic consecutive-topic swap and
    // WITHOUT relabelling to RankedReel. Exposed so DiversityReranker can order the very same set
    // with MMR (which subsumes the swap); also directly unit-testable.
    std::vector<Candidate> selectFeed(const User &user,
                                      const std::vector<Candidate> &rankedCandidates,
                                      std::size_t feedSize) const;

    // The scaled per-topic cap for a feed of `feedSize`: max(1, ceil(maxPerTopic * feedSize / 10)).
    // The TDD states the cap for a ten-item feed; this scales it proportionally so a 20-item feed
    // admits twice as many per topic and a tiny feed still admits at least one. Static + public so
    // the scaling rule is asserted directly in tests.
    static std::size_t topicCap(std::uint32_t maxPerTopic, std::size_t feedSize);

  private:
    const std::vector<Reel> &reels_;
    DiversityConfig config_;
};

} // namespace rr
