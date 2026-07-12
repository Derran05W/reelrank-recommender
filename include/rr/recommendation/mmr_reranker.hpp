#pragma once

#include <cstddef>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/reranker.hpp"

namespace rr {

// TDD 15.2 Maximum Marginal Relevance re-ranker (Phase 9, plan task 2).
//
// Pure MMR ORDERING — no constraint logic lives here (constraints are composed BEFORE MMR by
// DiversityReranker; see its header). Iteratively selects, from the remaining candidates, the one
// maximising
//
//     mmr(c) = lambda * relevance(c) - (1 - lambda) * maxSim(c, selected)
//
// and appends it to the feed, until feedSize items are chosen or the pool is exhausted.
//
//   * relevance(c): the candidate's rankingScore MIN-MAX normalised over the input pool to [0, 1],
//     so it shares the [0, 1]-ish scale of the cosine similarity term. DEGENERATE CASE — if every
//     rankingScore is equal (max == min, so min-max is undefined), relevance falls back to a
//     POSITION-BASED value relevance(c) = 1 - inputIndex / poolSize, which is strictly decreasing
//     in input position. This keeps the incoming relevance order meaningful (the pool arrives
//     relevance-sorted) and, at lambda == 1, reproduces the input order exactly. Documented and
//     deterministic.
//   * maxSim(c, selected): the maximum embedding cosine similarity (rr::dot on unit vectors)
//   between
//     c and any already-selected item. For the FIRST pick `selected` is empty and maxSim is defined
//     as 0 (no diversity penalty when nothing has been chosen yet).
//   * lambda: from DiversityConfig.mmrLambda (default 0.75). lambda == 1 => pure relevance order;
//     lambda == 0 => pure diversity after the (relevance-tie-broken) first pick.
//
// TIE-BREAKING. When two candidates have equal mmr(c), the higher rankingScore wins; if those tie
// too, the smaller ReelId wins. A total order, so selection is fully deterministic (no rr::Rng, no
// wall clock).
//
// OUTPUT. Each selected candidate becomes a RankedReel: reelId, score = candidate.rankingScore,
// rank = 0..n-1 selection position, sources = {candidate.source}, featureContributions copied from
// the candidate. (MMR reorders relevance; it never rewrites the score.)
//
// Ownership: NON-OWNING reference to the live reels vector for the dense-id embedding lookup
// reels[id.value]; a candidate whose reelId is out of range contributes a zero embedding (cosine 0)
// and is otherwise treated normally — the Orchestrator never sends such ids.
class MMRReranker final : public Reranker {
  public:
    MMRReranker(const std::vector<Reel> &reels, double lambda);

    std::vector<RankedReel> rerank(const User &user, const std::vector<Candidate> &rankedCandidates,
                                   std::size_t feedSize) const override;

  private:
    const std::vector<Reel> &reels_;
    double lambda_;
};

} // namespace rr
