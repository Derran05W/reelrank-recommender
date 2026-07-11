// Unit + property tests for PopularCandidateSource (Phase 6, TDD 12.3 / 13). Verify the
// deterministic order (smoothed-popularity desc, ties by ascending ReelId), the count cap,
// inactive/empty-embedding exclusion, the all-zero cold-start id fallback, the Candidate field
// contract (real cosine similarity + D3-inverse distance), and determinism.
#include "rr/candidate_sources/popular_candidate_source.hpp"

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
#include "rr/infrastructure/random.hpp"

namespace {

rr::Reel makeReel(uint32_t id, uint32_t creatorId, rr::Embedding emb, uint64_t imp, uint64_t comp,
                  uint64_t like = 0, uint64_t share = 0, bool active = true) {
    rr::Reel reel{};
    reel.id = rr::ReelId{id};
    reel.creatorId = rr::CreatorId{creatorId};
    rr::normalize(emb);
    reel.embedding = std::move(emb);
    reel.active = active;
    reel.impressionCount = imp;
    reel.completionCount = comp;
    reel.likeCount = like;
    reel.shareCount = share;
    return reel;
}

rr::User makeUser(rr::Embedding pref) {
    rr::User user{};
    user.id = rr::UserId{0};
    rr::normalize(pref);
    user.estimatedPreference = std::move(pref);
    return user;
}

std::vector<rr::ReelId> ids(const std::vector<rr::Candidate> &cands) {
    std::vector<rr::ReelId> out;
    for (const rr::Candidate &c : cands) {
        out.push_back(c.reelId);
    }
    return out;
}

// A non-degenerate embedding for property fuzzing: the first component is >= 1, so the vector is
// never near-zero and rr::normalize never throws.
rr::Embedding randEmbedding(rr::Rng &rng) {
    return {1.0f + static_cast<float>(std::fabs(rng.gaussian())),
            static_cast<float>(rng.gaussian()), static_cast<float>(rng.gaussian())};
}

// priorMean = 150/300 = 0.5; scores = (eng + 20*0.5)/(1 + 100 + 20) = (eng + 10)/121.
// reel0 eng=90 -> 100/121; reel2 eng=50 -> 60/121; reel1 eng=10 -> 20/121.
std::vector<rr::Reel> scoredReels() {
    return {makeReel(0, 100, {1.0f, 0.0f, 0.0f}, 100, 90),
            makeReel(1, 101, {0.0f, 1.0f, 0.0f}, 100, 10),
            makeReel(2, 102, {0.0f, 0.0f, 1.0f}, 100, 50)};
}

} // namespace

TEST(PopularCandidateSourceTest, OrdersByScoreDescending) {
    std::vector<rr::Reel> reels = scoredReels();
    rr::PopularCandidateSource source(reels, /*count=*/5);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 3u);
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{0}, rr::ReelId{2}, rr::ReelId{1}}));
    for (const rr::Candidate &c : cands) {
        EXPECT_EQ(c.source, rr::CandidateSource::Popular);
    }
}

TEST(PopularCandidateSourceTest, HonoursCountCap) {
    std::vector<rr::Reel> reels = scoredReels();
    rr::PopularCandidateSource source(reels, /*count=*/2);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 2u);
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{0}, rr::ReelId{2}}));
}

TEST(PopularCandidateSourceTest, ZeroCountReturnsEmpty) {
    std::vector<rr::Reel> reels = scoredReels();
    rr::PopularCandidateSource source(reels, /*count=*/0);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});
    EXPECT_TRUE(source.generate(user, rr::RecommendationRequest{}).empty());
}

TEST(PopularCandidateSourceTest, ExcludesInactiveAndEmptyEmbedding) {
    std::vector<rr::Reel> reels = scoredReels();
    reels[0].active = false;    // top scorer, but inactive
    reels[1].embedding.clear(); // invalid embedding
    rr::PopularCandidateSource source(reels, /*count=*/10);
    const rr::User user = makeUser({0.0f, 0.0f, 1.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 1u); // only reel2 survives
    EXPECT_EQ(cands.front().reelId, rr::ReelId{2});
}

// All-zero engagement => every smoothed score is 0 => the tie-break yields the lowest ReelIds,
// regardless of position in the reels vector (proves the tie-break is by id, not index).
TEST(PopularCandidateSourceTest, ColdStartFallsBackToLowestIds) {
    std::vector<rr::Reel> reels{
        makeReel(5, 100, {1.0f, 0.0f, 0.0f}, 0, 0), makeReel(2, 101, {0.0f, 1.0f, 0.0f}, 0, 0),
        makeReel(8, 102, {0.0f, 0.0f, 1.0f}, 0, 0), makeReel(1, 103, {1.0f, 1.0f, 0.0f}, 0, 0)};
    rr::PopularCandidateSource source(reels, /*count=*/2);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 2u);
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{1}, rr::ReelId{2}}));
}

// Candidate field contract: similarity is the real cosine (== dot for unit vectors) and distance
// is exactly the D3 inverse sqrt(2 - 2*sim).
TEST(PopularCandidateSourceTest, FillsSimilarityAndDistancePerD3) {
    std::vector<rr::Reel> reels = scoredReels();
    rr::PopularCandidateSource source(reels, /*count=*/5);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 3u);
    for (const rr::Candidate &c : cands) {
        const rr::Reel &reel = reels[c.reelId.value]; // ids equal indices in scoredReels()
        const float expectedSim = rr::dot(user.estimatedPreference, reel.embedding);
        EXPECT_NEAR(c.retrievalSimilarity, expectedSim, 1e-5f);
        const float expectedDist = std::sqrt(std::max(0.0f, 2.0f - 2.0f * c.retrievalSimilarity));
        EXPECT_FLOAT_EQ(c.retrievalDistance, expectedDist);
        EXPECT_EQ(c.rankingScore, 0.0f);
    }
    // reel0 == query => sim 1, distance 0.
    EXPECT_NEAR(cands.front().retrievalSimilarity, 1.0f, 1e-5f);
    EXPECT_NEAR(cands.front().retrievalDistance, 0.0f, 1e-3f);
}

TEST(PopularCandidateSourceTest, DeterministicAcrossCalls) {
    std::vector<rr::Reel> reels = scoredReels();
    rr::PopularCandidateSource source(reels, /*count=*/3);
    const rr::User user = makeUser({0.5f, 0.5f, 0.5f});

    const std::vector<rr::Candidate> a = source.generate(user, rr::RecommendationRequest{});
    const std::vector<rr::Candidate> b = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].reelId, b[i].reelId);
        EXPECT_EQ(a[i].retrievalSimilarity, b[i].retrievalSimilarity);
    }
}

// Property: over many random fixtures the result NEVER exceeds the configured count.
TEST(PopularCandidateSourceTest, NeverExceedsCountProperty) {
    for (uint64_t seed = 0; seed < 25; ++seed) {
        rr::Rng rng(seed);
        const uint32_t reelCount = 1 + static_cast<uint32_t>(rng.uniformInt(60));
        std::vector<rr::Reel> reels;
        reels.reserve(reelCount);
        for (uint32_t i = 0; i < reelCount; ++i) {
            const bool active = rng.bernoulli(0.8);
            reels.push_back(makeReel(i, 100 + i, randEmbedding(rng),
                                     static_cast<uint64_t>(rng.uniformInt(500)),
                                     static_cast<uint64_t>(rng.uniformInt(400)), 0, 0, active));
        }
        const uint32_t count = static_cast<uint32_t>(rng.uniformInt(30));
        rr::PopularCandidateSource source(reels, count);
        const rr::User user = makeUser(randEmbedding(rng));

        const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
        EXPECT_LE(cands.size(), static_cast<std::size_t>(count)) << "seed " << seed;
    }
}
