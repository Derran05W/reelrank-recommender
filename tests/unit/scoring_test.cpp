// Unit tests for the scaffolded Phase 6 scoring inputs (scoring.hpp / scoring.cpp): freshness
// decay, trending decay factor, trending velocity, and the popularity prior mean. Every case is a
// hand-computed golden against the documented header contract (TDD 8.2, 12.3, 12.4). These
// functions are pure and deterministic (D8/D9), so exact equalities are used where the arithmetic
// is exact in double.
#include "rr/recommendation/scoring.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "rr/domain/reel.hpp"
#include "rr/infrastructure/clock.hpp"

namespace {

// A reel carrying only the counters the scoring functions read; everything else is default.
rr::Reel makeReel(uint64_t impressions, uint64_t completions, uint64_t likes, uint64_t shares) {
    rr::Reel reel{};
    reel.impressionCount = impressions;
    reel.completionCount = completions;
    reel.likeCount = likes;
    reel.shareCount = shares;
    return reel;
}

// A reel carrying only the trending accumulator state.
rr::Reel trendingReel(double engagement, double impressions, rr::Timestamp updatedAt) {
    rr::Reel reel{};
    reel.trendingEngagement = engagement;
    reel.trendingImpressions = impressions;
    reel.trendingUpdatedAt = updatedAt;
    return reel;
}

constexpr double kFreshnessHalfLife = 604800.0; // 7 simulated days (config default)
constexpr double kTrendingHalfLife = 21600.0;   // 6 simulated hours (config default)

} // namespace

// --- freshnessScore ---------------------------------------------------------------------------

TEST(ScoringFreshnessTest, IsOneAtAgeZero) {
    EXPECT_DOUBLE_EQ(rr::freshnessScore(/*createdAt=*/100, /*now=*/100, kFreshnessHalfLife), 1.0);
}

TEST(ScoringFreshnessTest, HalvesExactlyAtOneHalfLife) {
    EXPECT_DOUBLE_EQ(rr::freshnessScore(/*createdAt=*/0, /*now=*/604800, kFreshnessHalfLife), 0.5);
}

TEST(ScoringFreshnessTest, QuartersAtTwoHalfLives) {
    EXPECT_DOUBLE_EQ(rr::freshnessScore(/*createdAt=*/0, /*now=*/1209600, kFreshnessHalfLife),
                     0.25);
}

TEST(ScoringFreshnessTest, FutureReelClampsToOne) {
    // now < createdAt (only possible in hand-built tests): age clamps to 0, score is 1.0.
    EXPECT_DOUBLE_EQ(rr::freshnessScore(/*createdAt=*/200, /*now=*/100, kFreshnessHalfLife), 1.0);
}

// --- trendingDecayFactor ----------------------------------------------------------------------

TEST(ScoringDecayTest, ClampsToOneWhenNowEqualsUpdatedAt) {
    EXPECT_DOUBLE_EQ(rr::trendingDecayFactor(/*updatedAt=*/1000, /*now=*/1000, kTrendingHalfLife),
                     1.0);
}

TEST(ScoringDecayTest, ClampsToOneWhenNowBeforeUpdatedAt) {
    EXPECT_DOUBLE_EQ(rr::trendingDecayFactor(/*updatedAt=*/1000, /*now=*/500, kTrendingHalfLife),
                     1.0);
}

TEST(ScoringDecayTest, HalvesExactlyAtOneHalfLife) {
    EXPECT_DOUBLE_EQ(rr::trendingDecayFactor(/*updatedAt=*/0, /*now=*/21600, kTrendingHalfLife),
                     0.5);
}

TEST(ScoringDecayTest, UnderflowsCleanlyForEnormousDt) {
    // dt/halfLife well past the double denormal floor (~1074): underflows to exactly 0, never
    // NaN/Inf.
    const double w = rr::trendingDecayFactor(/*updatedAt=*/0, /*now=*/100000, /*halfLife=*/1.0);
    EXPECT_TRUE(std::isfinite(w));
    EXPECT_DOUBLE_EQ(w, 0.0);
}

// --- trendingScore ----------------------------------------------------------------------------

TEST(ScoringTrendingTest, NeverInteractedReelScoresExactlyZero) {
    const rr::Reel reel = trendingReel(/*engagement=*/0.0, /*impressions=*/0.0, /*updatedAt=*/0);
    EXPECT_DOUBLE_EQ(rr::trendingScore(reel, /*now=*/50000, kTrendingHalfLife), 0.0);
}

TEST(ScoringTrendingTest, UndecayedScoreMatchesHandComputation) {
    // now == updatedAt => w = 1: trend = 1*8 / (1 + 1*4) = 8/5 = 1.6.
    const rr::Reel reel = trendingReel(/*engagement=*/8.0, /*impressions=*/4.0, /*updatedAt=*/1000);
    EXPECT_DOUBLE_EQ(rr::trendingScore(reel, /*now=*/1000, kTrendingHalfLife), 1.6);
}

TEST(ScoringTrendingTest, DecayedScoreMatchesHandComputation) {
    // one half-life => w = 0.5: trend = 0.5*8 / (1 + 0.5*4) = 4 / 3.
    const rr::Reel reel = trendingReel(/*engagement=*/8.0, /*impressions=*/4.0, /*updatedAt=*/0);
    EXPECT_DOUBLE_EQ(rr::trendingScore(reel, /*now=*/21600, kTrendingHalfLife), 4.0 / 3.0);
}

// --- engagementPriorMean ----------------------------------------------------------------------

TEST(ScoringPriorMeanTest, MatchesHandComputation) {
    // reel0: 3 + 2*5 + 4*7 = 41 engagement over 100 impressions.
    // reel1: 10           = 10 engagement over 100 impressions.
    // mean = (41 + 10) / (100 + 100) = 51 / 200.
    const std::vector<rr::Reel> reels{makeReel(100, 3, 5, 7), makeReel(100, 10, 0, 0)};
    EXPECT_DOUBLE_EQ(rr::engagementPriorMean(reels), 51.0 / 200.0);
}

TEST(ScoringPriorMeanTest, ZeroImpressionsReturnsZero) {
    const std::vector<rr::Reel> reels{makeReel(0, 0, 0, 0), makeReel(0, 0, 0, 0)};
    EXPECT_DOUBLE_EQ(rr::engagementPriorMean(reels), 0.0);
}

TEST(ScoringPriorMeanTest, EmptyCatalogReturnsZero) {
    EXPECT_DOUBLE_EQ(rr::engagementPriorMean(std::vector<rr::Reel>{}), 0.0);
}
