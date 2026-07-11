// Property tests for the WeightedRanker (Phase 6, TDD 14.2/24.3), seed-swept over many randomized
// fixture pools built with rr::Rng (in the TEST only - the ranker itself is pure/deterministic).
// Properties: (a) output rankingScore is monotonically non-increasing; (b) each candidate's
// contributions sum to its rankingScore to float tolerance; (c) the output is a permutation of the
// input pool.
#include "rr/recommendation/weighted_ranker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"

namespace {

constexpr int kNumSeeds = 32; // >= 20 randomized fixtures required by the phase plan.

rr::InteractionType randomType(rr::Rng &rng) {
    const rr::InteractionType types[] = {
        rr::InteractionType::Impression,   rr::InteractionType::InstantSkip,
        rr::InteractionType::PartialWatch, rr::InteractionType::CompleteWatch,
        rr::InteractionType::Rewatch,      rr::InteractionType::Like,
        rr::InteractionType::Share,        rr::InteractionType::FollowCreator,
        rr::InteractionType::NotInterested};
    return types[rng.uniformInt(9)];
}

// A random but self-consistent fixture: reels with dense ids, a user with affinities + history,
// and a candidate pool of unique reel ids with cosine similarities in [-1, 1].
struct Fixture {
    std::vector<rr::Reel> reels;
    rr::User user;
    std::vector<rr::Candidate> pool;
};

Fixture makeFixture(rr::Rng &rng) {
    Fixture fx;
    const std::size_t reelCount = 20 + rng.uniformInt(40); // 20..59
    const uint32_t creators = 6;
    const uint32_t topics = 5;
    fx.reels.reserve(reelCount);
    for (std::size_t i = 0; i < reelCount; ++i) {
        rr::Reel r{};
        r.id = rr::ReelId{static_cast<uint32_t>(i)};
        r.creatorId = rr::CreatorId{static_cast<uint32_t>(rng.uniformInt(creators))};
        r.primaryTopic = rr::TopicId{static_cast<uint32_t>(rng.uniformInt(topics))};
        // Random 2-D unit embedding (angle) so session_topic varies per candidate and the real
        // rr::dot path is exercised, not just the neutral fallback.
        const double reelTheta = rng.uniform(0.0, 6.283185307179586);
        r.embedding = {static_cast<float>(std::cos(reelTheta)),
                       static_cast<float>(std::sin(reelTheta))};
        r.intrinsicQuality = static_cast<float>(rng.uniform01());
        r.durationSeconds = static_cast<float>(rng.uniform(5.0, 120.0));
        r.createdAt = rng.uniformInt(2'000'000);
        r.impressionCount = rng.uniformInt(5000);
        r.completionCount = rng.uniformInt(r.impressionCount + 1);
        r.likeCount = rng.uniformInt(r.completionCount + 1);
        r.shareCount = rng.uniformInt(r.likeCount + 1);
        r.active = true;
        if (rng.bernoulli(0.6)) {
            r.trendingUpdatedAt = rng.uniformInt(2'000'000);
            r.trendingEngagement = rng.uniform(0.0, 50.0);
            r.trendingImpressions = rng.uniform(0.0, 200.0);
        }
        fx.reels.push_back(std::move(r));
    }

    // Random 2-D unit session preference (angle) so the session_topic feature is non-degenerate.
    const double sessionTheta = rng.uniform(0.0, 6.283185307179586);
    fx.user.sessionPreference = {static_cast<float>(std::cos(sessionTheta)),
                                 static_cast<float>(std::sin(sessionTheta))};

    for (uint32_t cr = 0; cr < creators; ++cr) {
        if (rng.bernoulli(0.5)) {
            fx.user.creatorAffinity[rr::CreatorId{cr}] = static_cast<float>(rng.uniform01());
        }
    }
    const std::size_t history = rng.uniformInt(20);
    for (std::size_t h = 0; h < history; ++h) {
        rr::InteractionEvent e{};
        e.reelId = rr::ReelId{static_cast<uint32_t>(rng.uniformInt(reelCount))};
        e.creatorId = fx.reels[e.reelId.value].creatorId;
        e.type = randomType(rng);
        fx.user.recentInteractions.push_back(e);
    }

    // Unique candidate ids: shuffle-free selection via a random subset of reel indices.
    std::vector<uint32_t> ids(reelCount);
    for (std::size_t i = 0; i < reelCount; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }
    for (std::size_t i = reelCount; i > 1; --i) { // Fisher-Yates with rr::Rng
        const std::size_t j = rng.uniformInt(i);
        std::swap(ids[i - 1], ids[j]);
    }
    const std::size_t poolSize = 5 + rng.uniformInt(reelCount - 5 + 1);
    fx.pool.reserve(poolSize);
    for (std::size_t i = 0; i < poolSize; ++i) {
        rr::Candidate c{};
        c.reelId = rr::ReelId{ids[i]};
        c.source = rr::CandidateSource::VectorHNSW;
        c.retrievalSimilarity = static_cast<float>(rng.uniform(-1.0, 1.0));
        fx.pool.push_back(c);
    }
    return fx;
}

} // namespace

TEST(RankingPropertyTest, MonotoneSumAndPermutation) {
    for (int seed = 0; seed < kNumSeeds; ++seed) {
        rr::Rng rng(static_cast<uint64_t>(1000 + seed));
        const Fixture fx = makeFixture(rng);
        rr::WeightedRanker ranker(fx.reels, {});
        const std::vector<rr::Candidate> ranked = ranker.rank(fx.user, fx.pool, 1'500'000);

        // (c) permutation of the input pool.
        ASSERT_EQ(ranked.size(), fx.pool.size()) << "seed " << seed;
        std::vector<uint32_t> inIds;
        std::vector<uint32_t> outIds;
        for (const rr::Candidate &c : fx.pool) {
            inIds.push_back(c.reelId.value);
        }
        for (const rr::Candidate &c : ranked) {
            outIds.push_back(c.reelId.value);
        }
        std::sort(inIds.begin(), inIds.end());
        std::sort(outIds.begin(), outIds.end());
        EXPECT_EQ(inIds, outIds) << "seed " << seed << ": output is not a permutation of input";

        for (std::size_t i = 0; i < ranked.size(); ++i) {
            // (a) monotonically non-increasing score.
            if (i > 0) {
                EXPECT_GE(ranked[i - 1].rankingScore, ranked[i].rankingScore)
                    << "seed " << seed << " position " << i;
            }
            // (b) contributions sum to the score.
            const auto &m = ranked[i].featureContributions;
            EXPECT_EQ(m.size(), 11u) << "seed " << seed;
            double sum = 0.0;
            for (const auto &[key, value] : m) {
                sum += value;
            }
            const double score = ranked[i].rankingScore;
            const double absErr = std::fabs(sum - score);
            const double relOk = absErr <= 1e-4 * std::max(1.0, std::fabs(score));
            EXPECT_TRUE(absErr <= 1e-6 || relOk)
                << "seed " << seed << " position " << i << ": sum " << sum << " != score " << score;
        }
    }
}
