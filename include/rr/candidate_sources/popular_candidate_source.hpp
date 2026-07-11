#pragma once

#include <cstdint>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"

namespace rr {

// TDD 12.3 popular candidate source: the top reels by Bayesian-smoothed engagement, exposed as a
// CandidateGenerator for the orchestrated pipeline (TDD 13). Scores every active reel with
// rr::smoothedPopularity against the live global prior mean (rr::engagementPriorMean) recomputed
// each request, so the ranking tracks the simulation. The order is a fully deterministic TOTAL
// order: score descending, ties broken by ascending ReelId. Cold start (no engagement anywhere)
// makes every score 0, so the tie-break yields the first `count` active reels by id — the same
// documented semantics as PopularityRecommender (contrast TrendingCandidateSource, which returns
// nothing when nothing is trending).
//
// COST: O(catalog) per request (one prior-mean pass + one scoring pass + a partial_sort selecting
// the top `count`). Accepted for Phase 6 (TDD 13); the constant factor is kept tight via a reused
// scratch buffer with no per-reel allocation.
//
// FILTERING: skips inactive reels and reels with empty embeddings cheaply during the scan, but
// does NOT seen-filter — dedup / seen-removal / pool-cap are the Orchestrator's job (TDD 13).
//
// CANDIDATE FIELDS: filled exactly like the vector sources — reelId, source=Popular, and
// retrievalSimilarity = cos(effectivePreference(user), reel.embedding) (unit vectors, D3) with
// retrievalDistance = sqrt(max(0, 2 - 2*sim)) the D3 inverse. The similarity is REAL, not a fake
// 0: the Orchestrator's pool cap orders by retrievalSimilarity and the ranker consumes it as its
// similarity feature, so a 0 here would starve popular candidates at the cap.
class PopularCandidateSource final : public CandidateGenerator {
  public:
    PopularCandidateSource(const std::vector<Reel> &reels, uint32_t count);

    std::vector<Candidate> generate(const User &user,
                                    const RecommendationRequest &request) override;

  private:
    struct Scored {
        double score;
        uint32_t index; // index into reels_
    };

    const std::vector<Reel> &reels_;
    uint32_t count_;
    std::vector<Scored> scratch_; // reused across generate() calls (single-threaded core, D13)
};

} // namespace rr
