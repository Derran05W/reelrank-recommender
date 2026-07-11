// Unit + property tests for CreatorAffinityCandidateSource (Phase 6, TDD 12.6 / 13). Verify the
// affinity*smoothed-engagement ordering (ties by ascending ReelId), that only creators with
// affinity > 0 contribute, the empty-affinity => empty-result contract, the count cap,
// inactive/empty-embedding exclusion, the Candidate field contract, and determinism.
#include "rr/candidate_sources/creator_affinity_candidate_source.hpp"

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
                  bool active = true) {
    rr::Reel reel{};
    reel.id = rr::ReelId{id};
    reel.creatorId = rr::CreatorId{creatorId};
    rr::normalize(emb);
    reel.embedding = std::move(emb);
    reel.active = active;
    reel.impressionCount = imp;
    reel.completionCount = comp;
    return reel;
}

rr::User makeUser(rr::Embedding pref, const std::vector<std::pair<uint32_t, float>> &affinities) {
    rr::User user{};
    user.id = rr::UserId{0};
    rr::normalize(pref);
    user.estimatedPreference = std::move(pref);
    for (const auto &[creatorId, affinity] : affinities) {
        user.creatorAffinity[rr::CreatorId{creatorId}] = affinity;
    }
    return user;
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

// smoothed engagement (zero prior) = comp / (1 + imp + 20).
// reel0 creator10: 100/121;  reel1 creator10: 0;  reel2 creator20: 50/121;  reel3 creator20: 0;
// reel4 creator30: 80/121 (only surfaces if creator30 has affinity).
std::vector<rr::Reel> catalogReels() {
    return {makeReel(0, 10, {1.0f, 0.0f, 0.0f}, 100, 100),
            makeReel(1, 10, {0.0f, 1.0f, 0.0f}, 100, 0),
            makeReel(2, 20, {0.0f, 0.0f, 1.0f}, 100, 50), makeReel(3, 20, {1.0f, 1.0f, 0.0f}, 0, 0),
            makeReel(4, 30, {1.0f, 0.0f, 1.0f}, 100, 80)};
}

} // namespace

// score = affinity * smoothed engagement. With {10:0.8, 20:0.5}:
// reel0 0.8*100/121=0.661; reel2 0.5*50/121=0.207; reel1 & reel3 score 0 -> id tie-break.
// creator30 (reel4) has no affinity, so it never appears.
TEST(CreatorAffinityCandidateSourceTest, OrdersByAffinityTimesEngagement) {
    std::vector<rr::Reel> reels = catalogReels();
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/10);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {{10, 0.8f}, {20, 0.5f}});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 4u); // reel4 (creator 30, no affinity) excluded
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{0}, rr::ReelId{2}, rr::ReelId{1},
                                                   rr::ReelId{3}}));
    for (const rr::Candidate &c : cands) {
        EXPECT_EQ(c.source, rr::CandidateSource::CreatorAffinity);
        EXPECT_NE(c.reelId, rr::ReelId{4});
    }
}

TEST(CreatorAffinityCandidateSourceTest, HonoursCountCap) {
    std::vector<rr::Reel> reels = catalogReels();
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/2);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {{10, 0.8f}, {20, 0.5f}});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 2u);
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{0}, rr::ReelId{2}}));
}

TEST(CreatorAffinityCandidateSourceTest, EmptyAffinityReturnsEmpty) {
    std::vector<rr::Reel> reels = catalogReels();
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/10);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {}); // no creatorAffinity entries
    EXPECT_TRUE(source.generate(user, rr::RecommendationRequest{}).empty());
}

// A zero (or negative) affinity does not qualify a creator; only creator 20 contributes here.
TEST(CreatorAffinityCandidateSourceTest, NonPositiveAffinityCreatorsExcluded) {
    std::vector<rr::Reel> reels = catalogReels();
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/10);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {{10, 0.0f}, {20, 0.5f}});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 2u); // only creator 20's reels (2 and 3)
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{2}, rr::ReelId{3}}));
}

TEST(CreatorAffinityCandidateSourceTest, ZeroCountReturnsEmpty) {
    std::vector<rr::Reel> reels = catalogReels();
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/0);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {{10, 0.8f}});
    EXPECT_TRUE(source.generate(user, rr::RecommendationRequest{}).empty());
}

TEST(CreatorAffinityCandidateSourceTest, ExcludesInactiveAndEmptyEmbedding) {
    std::vector<rr::Reel> reels = catalogReels();
    reels[0].active = false;    // creator 10's top scorer, inactive
    reels[2].embedding.clear(); // creator 20's scorer, invalid embedding
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/10);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {{10, 0.8f}, {20, 0.5f}});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(cands.size(), 2u); // reel1 (creator10) and reel3 (creator20), both score 0
    EXPECT_EQ(ids(cands), (std::vector<rr::ReelId>{rr::ReelId{1}, rr::ReelId{3}}));
}

TEST(CreatorAffinityCandidateSourceTest, FillsSimilarityAndDistancePerD3) {
    std::vector<rr::Reel> reels = catalogReels();
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/10);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {{10, 0.8f}, {20, 0.5f}});

    const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
    ASSERT_FALSE(cands.empty());
    for (const rr::Candidate &c : cands) {
        const rr::Reel &reel = reels[c.reelId.value]; // ids equal indices in catalogReels()
        const float expectedSim = rr::dot(user.estimatedPreference, reel.embedding);
        EXPECT_NEAR(c.retrievalSimilarity, expectedSim, 1e-5f);
        const float expectedDist = std::sqrt(std::max(0.0f, 2.0f - 2.0f * c.retrievalSimilarity));
        EXPECT_FLOAT_EQ(c.retrievalDistance, expectedDist);
        EXPECT_EQ(c.rankingScore, 0.0f);
    }
}

TEST(CreatorAffinityCandidateSourceTest, DeterministicAcrossCalls) {
    std::vector<rr::Reel> reels = catalogReels();
    rr::CreatorAffinityCandidateSource source(reels, /*count=*/10);
    const rr::User user = makeUser({0.5f, 0.5f, 0.5f}, {{10, 0.8f}, {20, 0.5f}});

    const std::vector<rr::Candidate> a = source.generate(user, rr::RecommendationRequest{});
    const std::vector<rr::Candidate> b = source.generate(user, rr::RecommendationRequest{});
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].reelId, b[i].reelId);
        EXPECT_EQ(a[i].retrievalSimilarity, b[i].retrievalSimilarity);
    }
}

// Property: over many random fixtures the result NEVER exceeds the configured count.
TEST(CreatorAffinityCandidateSourceTest, NeverExceedsCountProperty) {
    constexpr uint32_t kCreators = 5;
    for (uint64_t seed = 0; seed < 25; ++seed) {
        rr::Rng rng(seed);
        const uint32_t reelCount = 1 + static_cast<uint32_t>(rng.uniformInt(60));
        std::vector<rr::Reel> reels;
        reels.reserve(reelCount);
        for (uint32_t i = 0; i < reelCount; ++i) {
            const bool active = rng.bernoulli(0.8);
            reels.push_back(makeReel(i, 100 + (i % kCreators), randEmbedding(rng),
                                     static_cast<uint64_t>(rng.uniformInt(500)),
                                     static_cast<uint64_t>(rng.uniformInt(400)), active));
        }
        std::vector<std::pair<uint32_t, float>> affinities;
        for (uint32_t c = 0; c < kCreators; ++c) {
            if (rng.bernoulli(0.6)) {
                affinities.emplace_back(100 + c, static_cast<float>(rng.uniform01()));
            }
        }
        const uint32_t count = static_cast<uint32_t>(rng.uniformInt(30));
        rr::CreatorAffinityCandidateSource source(reels, count);
        const rr::User user = makeUser(randEmbedding(rng), affinities);

        const std::vector<rr::Candidate> cands = source.generate(user, rr::RecommendationRequest{});
        EXPECT_LE(cands.size(), static_cast<std::size_t>(count)) << "seed " << seed;
    }
}
