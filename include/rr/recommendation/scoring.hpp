#pragma once

#include <vector>

#include "rr/domain/reel.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// Shared time-sensitive scoring inputs for Phase 6+ (TDD 8.2, 12.3, 12.4). These free functions
// are the single source of truth for the freshness and trending math used by both the candidate
// sources and the ranking features, so the two can never drift apart. All of them are pure and
// deterministic (D8: no randomness, logical Timestamps only, D9).

// Freshness (TDD 8.2): exponential decay from upload age, 2^(-age/halfLife) in (0, 1].
// age = now - createdAt in simulated seconds; a reel "from the future" (now < createdAt, possible
// only in hand-built tests) clamps to age 0 and scores 1. halfLifeSeconds must be > 0 (config
// default 604800 = 7 simulated days, RankingConfig::freshnessHalfLifeSeconds).
double freshnessScore(Timestamp createdAt, Timestamp now, double halfLifeSeconds);

// The exponential decay factor w(dt) = 2^(-dt/halfLife) (TDD 12.4, expressed via half-life:
// e^(-lambda*dt) with lambda = ln2/halfLife) that brings an accumulator last updated at
// `updatedAt` forward to `now`. Clamps to 1 when now <= updatedAt. Underflows to 0 for huge dt.
double trendingDecayFactor(Timestamp updatedAt, Timestamp now, double halfLifeSeconds);

// Trending velocity (TDD 12.4): decayed weighted engagement over decayed impressions,
//   trend = w * reel.trendingEngagement / (1 + w * reel.trendingImpressions)
// where w = trendingDecayFactor(reel.trendingUpdatedAt, now, halfLifeSeconds). The accumulators
// themselves are maintained by Simulator::step (see the contract on rr::Reel); this read-side
// decay makes the score correct at any `now` without requiring an update per query. A reel with
// no interactions yet scores 0.
double trendingScore(const Reel &reel, Timestamp now, double halfLifeSeconds);

// Global mean engagement rate sum(popularityEngagement) / sum(impressionCount) over all reels,
// 0 before any impression exists. The Bayesian prior mean for smoothedPopularity (TDD 12.3),
// factored out of PopularityRecommender so the popular candidate source and the popularity
// ranking feature share it.
double engagementPriorMean(const std::vector<Reel> &reels);

} // namespace rr
