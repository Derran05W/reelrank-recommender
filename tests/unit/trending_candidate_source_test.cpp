// Unit + property tests for TrendingCandidateSource (Phase 6, TDD 12.4 / 13). Verify the
// positive-velocity-only qualification (and the empty cold-start contrast with Popular), the
// deterministic order (trending score desc, ties by ascending ReelId), read-side time decay, the
// count cap, inactive/empty-embedding exclusion, the Candidate field contract, and determinism.
#include "rr/candidate_sources/trending_candidate_source.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/scoring.hpp"

namespace {

constexpr double kHalfLife = 1000.0;

rr::Reel makeReel(uint32_t id, uint32_t creatorId, rr::Embedding emb, double trendingEngagement,
                  double trendingImpressions, rr::Timestamp trendingUpdatedAt, bool active = true) {
    rr::Reel reel{};
    reel.id = rr::ReelId{id};
    reel.creatorId = rr::CreatorId{creatorId};
    rr::normalize(emb);
    reel.embedding = std::move(emb);
    reel.active = active;
    reel.trendingEngagement = trendingEngagement;
    reel.trendingImpressions = trendingImpressions;
    reel.trendingUpdatedAt = trendingUpdatedAt;
    return reel;
}

rr::User makeUser(rr::Embedding pref) {
    rr::User user{};
    user.id = rr::UserId{0};
    rr::normalize(pref);
    user.estimatedPreference = std::move(pref);
    return user;
}

rr::RecommendationRequest request(rr::Timestamp requestTime) {
    rr::RecommendationRequest req{};
    req.requestTime = requestTime;
    return req;
}

std::vector<rr::ReelId> ids(const std::vector<rr::Candidate> &cands) {
    std::vector<rr::ReelId> out;
    for (const rr::Candidate &c : cands) {
        out.push_back(c.reelId);
    }
    return out;
}

rr::Embedding randEmbedding(rr::Rng &rng) {
    return {1.0f + static_cast<float>(std::fabs(rng.gaussian())),
            static_cast<float>(rng.gaussian()), static_cast<float>(rng.gaussian())};
}

// requestTime == trendingUpdatedAt == 0 => decay factor 1, so score = te / (1 + ti).
// reel3 6/3=2.0; reel0 10/6=1.667; reel1 2/6=0.333; reel2 0 (not trending, excluded).
std::vector<rr::Reel> velocityReels() {
    return {makeReel(0, 100, {1.0f, 0.0f, 0.0f}, 10.0, 5.0, 0),
            makeReel(1, 101, {0.0f, 1.0f, 0.0f}, 2.0, 5.0, 0),
            makeReel(2, 102, {0.0f, 0.0f, 1.0f}, 0.0, 0.0, 0),
            makeReel(3, 103, {1.0f, 1.0f, 0.0f}, 6.0, 2.0, 0)};
}

} // namespace

TEST(TrendingCandidateSourceTest, OrdersByScoreAndExcludesZeroVelocity) {
    std::vector<rr::Reel> reels = velocityReels();
    rr::TrendingCandidateSource source(reels, /*count=*/10, kHalfLife);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, request(0));
    ASSERT_EQ(cands.size(), 3u); // reel2 (zero velocity) excluded
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{3}, rr::ReelId{0}, rr::ReelId{1}}));
    for (const rr::Candidate &c : cands) {
        EXPECT_EQ(c.source, rr::CandidateSource::Trending);
    }
}

TEST(TrendingCandidateSourceTest, HonoursCountCap) {
    std::vector<rr::Reel> reels = velocityReels();
    rr::TrendingCandidateSource source(reels, /*count=*/2, kHalfLife);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, request(0));
    ASSERT_EQ(cands.size(), 2u);
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{3}, rr::ReelId{0}}));
}

// Contrast with Popular's cold-start id fallback: a catalog with no decayed interactions is not
// "trending", so the source returns NOTHING rather than the first reels by id.
TEST(TrendingCandidateSourceTest, ColdCatalogReturnsEmpty) {
    std::vector<rr::Reel> reels{makeReel(0, 100, {1.0f, 0.0f, 0.0f}, 0.0, 0.0, 0),
                                makeReel(1, 101, {0.0f, 1.0f, 0.0f}, 0.0, 0.0, 0)};
    rr::TrendingCandidateSource source(reels, /*count=*/10, kHalfLife);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});
    EXPECT_TRUE(source.generate(user, request(0)).empty());
}

TEST(TrendingCandidateSourceTest, ZeroCountReturnsEmpty) {
    std::vector<rr::Reel> reels = velocityReels();
    rr::TrendingCandidateSource source(reels, /*count=*/0, kHalfLife);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});
    EXPECT_TRUE(source.generate(user, request(0)).empty());
}

// Read-side time decay: two reels with identical accumulators but different last-update times.
// At requestTime the fresher reel (updated at requestTime) decays by 1; the stale one decays by
// 2^(-1) at one half-life, so it scores lower.
TEST(TrendingCandidateSourceTest, AppliesReadSideTimeDecay) {
    const rr::Timestamp t = 1000;
    std::vector<rr::Reel> reels{
        makeReel(0, 100, {1.0f, 0.0f, 0.0f}, 10.0, 5.0, /*updatedAt=*/t),  // fresh: w = 1
        makeReel(1, 101, {0.0f, 1.0f, 0.0f}, 10.0, 5.0, /*updatedAt=*/0)}; // stale: w = 0.5
    rr::TrendingCandidateSource source(reels, /*count=*/10, /*halfLife=*/1000.0);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, request(t));
    ASSERT_EQ(cands.size(), 2u);
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{0}, rr::ReelId{1}}));
    // Cross-check the decayed score of the stale reel against the scoring oracle.
    EXPECT_GT(rr::trendingScore(reels[0], t, 1000.0), rr::trendingScore(reels[1], t, 1000.0));
    EXPECT_NEAR(rr::trendingScore(reels[1], t, 1000.0), 0.5 * 10.0 / (1.0 + 0.5 * 5.0), 1e-12);
}

TEST(TrendingCandidateSourceTest, ExcludesInactiveAndEmptyEmbedding) {
    std::vector<rr::Reel> reels = velocityReels();
    reels[3].active = false;    // top scorer, inactive
    reels[0].embedding.clear(); // invalid embedding
    rr::TrendingCandidateSource source(reels, /*count=*/10, kHalfLife);
    const rr::User user = makeUser({0.0f, 1.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, request(0));
    ASSERT_EQ(cands.size(), 1u); // only reel1 remains trending + valid
    EXPECT_EQ(cands.front().reelId, rr::ReelId{1});
}

TEST(TrendingCandidateSourceTest, FillsSimilarityAndDistancePerD3) {
    std::vector<rr::Reel> reels = velocityReels();
    rr::TrendingCandidateSource source(reels, /*count=*/10, kHalfLife);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, request(0));
    ASSERT_FALSE(cands.empty());
    for (const rr::Candidate &c : cands) {
        const rr::Reel &reel = reels[c.reelId.value]; // ids equal indices in velocityReels()
        const float expectedSim = rr::dot(user.estimatedPreference, reel.embedding);
        EXPECT_NEAR(c.retrievalSimilarity, expectedSim, 1e-5f);
        const float expectedDist = std::sqrt(std::max(0.0f, 2.0f - 2.0f * c.retrievalSimilarity));
        EXPECT_FLOAT_EQ(c.retrievalDistance, expectedDist);
        EXPECT_EQ(c.rankingScore, 0.0f);
    }
}

TEST(TrendingCandidateSourceTest, DeterministicAcrossCalls) {
    std::vector<rr::Reel> reels = velocityReels();
    rr::TrendingCandidateSource source(reels, /*count=*/10, kHalfLife);
    const rr::User user = makeUser({0.5f, 0.5f, 0.5f});

    const std::vector<rr::Candidate> a = source.generate(user, request(0));
    const std::vector<rr::Candidate> b = source.generate(user, request(0));
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].reelId, b[i].reelId);
        EXPECT_EQ(a[i].retrievalSimilarity, b[i].retrievalSimilarity);
    }
}

// Property: over many random fixtures the result NEVER exceeds the configured count.
TEST(TrendingCandidateSourceTest, NeverExceedsCountProperty) {
    for (uint64_t seed = 0; seed < 25; ++seed) {
        rr::Rng rng(seed);
        const uint32_t reelCount = 1 + static_cast<uint32_t>(rng.uniformInt(60));
        std::vector<rr::Reel> reels;
        reels.reserve(reelCount);
        for (uint32_t i = 0; i < reelCount; ++i) {
            const bool active = rng.bernoulli(0.8);
            reels.push_back(makeReel(i, 100 + i, randEmbedding(rng), rng.uniform(0.0, 20.0),
                                     rng.uniform(0.0, 20.0),
                                     static_cast<rr::Timestamp>(rng.uniformInt(2000)), active));
        }
        const uint32_t count = static_cast<uint32_t>(rng.uniformInt(30));
        rr::TrendingCandidateSource source(reels, count, kHalfLife);
        const rr::User user = makeUser(randEmbedding(rng));

        const std::vector<rr::Candidate> cands =
            source.generate(user, request(static_cast<rr::Timestamp>(rng.uniformInt(3000))));
        EXPECT_LE(cands.size(), static_cast<std::size_t>(count)) << "seed " << seed;
    }
}
