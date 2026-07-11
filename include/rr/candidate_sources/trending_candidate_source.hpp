#pragma once

#include <cstdint>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"

namespace rr {

// TDD 12.4 trending candidate source: the top reels by recent-velocity score, exposed as a
// CandidateGenerator for the orchestrated pipeline (TDD 13). Scores every active reel with
// rr::trendingScore(reel, request.requestTime, halfLifeSeconds) — the decayed weighted engagement
// over decayed impressions maintained by Simulator::step — and returns the highest scorers in a
// fully deterministic TOTAL order: score descending, ties broken by ascending ReelId.
//
// Only reels with score > 0 QUALIFY: a zero-velocity reel (no decayed interactions) is not
// "trending". This is the deliberate contrast with PopularCandidateSource: on a cold catalog
// Popular falls back to the first reels by id, whereas Trending returns NOTHING (and in general
// may return FEWER than `count`), because surfacing the first N reels by id as "trending" would be
// meaningless.
//
// COST: O(catalog) per request (one scoring pass + a partial_sort of the qualifying reels).
// Accepted for Phase 6 (TDD 13); the constant factor is kept tight via a reused scratch buffer
// with no per-reel allocation.
//
// FILTERING: skips inactive reels and empty embeddings during the scan; dedup / seen / pool-cap
// are the Orchestrator's job (TDD 13). CANDIDATE FIELDS are filled exactly like the vector sources
// (reelId, source=Trending, retrievalSimilarity = cos(effectivePreference(user), reel.embedding),
// D3-inverse distance) — the real similarity matters at the Orchestrator's pool cap and as the
// ranker's similarity feature.
class TrendingCandidateSource final : public CandidateGenerator {
  public:
    TrendingCandidateSource(const std::vector<Reel> &reels, uint32_t count, double halfLifeSeconds);

    std::vector<Candidate> generate(const User &user,
                                    const RecommendationRequest &request) override;

  private:
    struct Scored {
        double score;
        uint32_t index; // index into reels_
    };

    const std::vector<Reel> &reels_;
    uint32_t count_;
    double halfLifeSeconds_;
    std::vector<Scored> scratch_; // reused across generate() calls (single-threaded core, D13)
};

} // namespace rr
