// Flagship diversity property suite (Phase 9, TDD 24.3 / plan task 5). Seed-swept over many
// randomized candidate pools + users, in BOTH composition modes (useMmr false/true), it asserts the
// invariant that owns the phase's exit criterion "duplicate/repetitive content eliminated": no feed
// the DiversityReranker produces ever contains a duplicate id, a seen reel, more than maxPerCreator
// per creator, or more than the scaled topic cap per primary topic. It also asserts the feed is
// FULL whenever a cap-feasible selection of that size exists (verified against an INDEPENDENT
// greedy replay), and that the reranker is deterministic.
#include "rr/recommendation/diversity_reranker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"

namespace {

constexpr int kNumSeeds = 24; // >= 20 randomized fixtures required by the phase plan.

struct Fixture {
    std::vector<rr::Reel> reels;
    rr::User user;
    std::vector<rr::Candidate> pool;
    rr::DiversityConfig config;
    std::size_t feedSize;
};

rr::CandidateSource randomSource(rr::Rng &rng) {
    const rr::CandidateSource sources[] = {
        rr::CandidateSource::VectorHNSW, rr::CandidateSource::Trending, rr::CandidateSource::Fresh,
        rr::CandidateSource::Popular, rr::CandidateSource::Exploration};
    return sources[rng.uniformInt(5)];
}

Fixture makeFixture(rr::Rng &rng) {
    Fixture fx;
    const std::size_t reelCount = 30 + rng.uniformInt(50);                  // 30..79
    const uint32_t creators = 3 + static_cast<uint32_t>(rng.uniformInt(6)); // 3..8
    const uint32_t topics = 3 + static_cast<uint32_t>(rng.uniformInt(5));   // 3..7
    fx.reels.reserve(reelCount);
    for (std::size_t i = 0; i < reelCount; ++i) {
        rr::Reel r{};
        r.id = rr::ReelId{static_cast<uint32_t>(i)};
        r.creatorId = rr::CreatorId{static_cast<uint32_t>(rng.uniformInt(creators))};
        r.primaryTopic = rr::TopicId{static_cast<uint32_t>(rng.uniformInt(topics))};
        const double theta = rng.uniform(0.0, 6.283185307179586);
        r.embedding = {static_cast<float>(std::cos(theta)), static_cast<float>(std::sin(theta))};
        r.active = true;
        fx.reels.push_back(std::move(r));
        if (rng.bernoulli(0.2)) {
            fx.user.seenReels.insert(rr::ReelId{static_cast<uint32_t>(i)});
        }
    }

    // A pool of mostly-unique reel ids in random (relevance) order, with occasional duplicate ids
    // injected to exercise the no-duplicate rule (the Orchestrator dedups upstream, but the
    // reranker must be robust on its own).
    std::vector<uint32_t> ids(reelCount);
    for (std::size_t i = 0; i < reelCount; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }
    for (std::size_t i = reelCount; i > 1; --i) { // Fisher-Yates with rr::Rng
        const std::size_t j = rng.uniformInt(i);
        std::swap(ids[i - 1], ids[j]);
    }
    const std::size_t poolSize = 10 + rng.uniformInt(reelCount - 10 + 1);
    fx.pool.reserve(poolSize + 3);
    for (std::size_t i = 0; i < poolSize; ++i) {
        rr::Candidate c{};
        c.reelId = rr::ReelId{ids[i]};
        c.source = randomSource(rng);
        c.rankingScore = static_cast<float>(rng.uniform(-1.0, 1.0));
        fx.pool.push_back(c);
    }
    for (int d = 0; d < 3; ++d) {
        if (rng.bernoulli(0.5) && !fx.pool.empty()) {
            rr::Candidate c = fx.pool[rng.uniformInt(fx.pool.size())]; // duplicate id, fresh score
            c.rankingScore = static_cast<float>(rng.uniform(-1.0, 1.0));
            fx.pool.push_back(c);
        }
    }

    fx.config.enabled = true;
    fx.config.maxPerCreator = 1 + static_cast<uint32_t>(rng.uniformInt(3)); // 1..3
    fx.config.maxPerTopic = 1 + static_cast<uint32_t>(rng.uniformInt(4));   // 1..4
    fx.config.mmrLambda = rng.uniform(0.0, 1.0);
    fx.feedSize = 5 + rng.uniformInt(16); // 5..20
    return fx;
}

// INDEPENDENT replay of the hard-rule greedy walk (mirrors ConstraintReranker but recomputed here,
// including its own topic-cap formula, so it is a genuine cross-check not a call into the SUT).
// Returns the admitted reel ids in walk order.
std::vector<uint32_t> greedyReplay(const Fixture &fx) {
    const std::size_t topicLimit = static_cast<std::size_t>(
        std::max(1.0, std::ceil(static_cast<double>(fx.config.maxPerTopic) *
                                static_cast<double>(fx.feedSize) / 10.0)));
    std::vector<uint32_t> admitted;
    std::unordered_set<uint32_t> chosen;
    std::unordered_map<uint32_t, uint32_t> creatorCounts;
    std::unordered_map<uint32_t, std::size_t> topicCounts;
    for (const rr::Candidate &c : fx.pool) {
        if (admitted.size() >= fx.feedSize) {
            break;
        }
        const uint32_t id = c.reelId.value;
        if (id >= fx.reels.size() || chosen.count(id) != 0 ||
            fx.user.seenReels.contains(rr::ReelId{id})) {
            continue;
        }
        const rr::Reel &reel = fx.reels[id];
        if (creatorCounts[reel.creatorId.value] >= fx.config.maxPerCreator) {
            continue;
        }
        if (topicCounts[reel.primaryTopic.value] >= topicLimit) {
            continue;
        }
        chosen.insert(id);
        ++creatorCounts[reel.creatorId.value];
        ++topicCounts[reel.primaryTopic.value];
        admitted.push_back(id);
    }
    return admitted;
}

std::vector<uint32_t> feedIds(const std::vector<rr::RankedReel> &feed) {
    std::vector<uint32_t> ids;
    for (const rr::RankedReel &r : feed) {
        ids.push_back(r.reelId.value);
    }
    return ids;
}

std::vector<uint32_t> sorted(std::vector<uint32_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

void checkInvariants(const Fixture &fx, const std::vector<rr::RankedReel> &feed, int seed,
                     bool useMmr) {
    const std::size_t topicLimit = static_cast<std::size_t>(
        std::max(1.0, std::ceil(static_cast<double>(fx.config.maxPerTopic) *
                                static_cast<double>(fx.feedSize) / 10.0)));

    std::unordered_set<uint32_t> seenInFeed;
    std::unordered_map<uint32_t, uint32_t> creatorCounts;
    std::unordered_map<uint32_t, std::size_t> topicCounts;
    for (const rr::RankedReel &r : feed) {
        const uint32_t id = r.reelId.value;
        // No duplicate ids.
        EXPECT_TRUE(seenInFeed.insert(id).second)
            << "seed " << seed << " useMmr " << useMmr << ": duplicate id " << id;
        // No seen reels.
        EXPECT_FALSE(fx.user.seenReels.contains(rr::ReelId{id}))
            << "seed " << seed << " useMmr " << useMmr << ": seen id " << id;
        ASSERT_LT(id, fx.reels.size());
        const rr::Reel &reel = fx.reels[id];
        ++creatorCounts[reel.creatorId.value];
        ++topicCounts[reel.primaryTopic.value];
    }
    for (const auto &[creator, count] : creatorCounts) {
        EXPECT_LE(count, fx.config.maxPerCreator)
            << "seed " << seed << " useMmr " << useMmr << ": creator " << creator << " over cap";
    }
    for (const auto &[topic, count] : topicCounts) {
        EXPECT_LE(count, topicLimit)
            << "seed " << seed << " useMmr " << useMmr << ": topic " << topic << " over cap";
    }
    EXPECT_LE(feed.size(), fx.feedSize) << "seed " << seed << " useMmr " << useMmr;

    // The feed SET matches the independent greedy replay; in particular the feed is FULL (==
    // feedSize) exactly when the greedy walk can admit that many, and shorter only when caps make
    // it infeasible.
    const std::vector<uint32_t> replay = greedyReplay(fx);
    EXPECT_EQ(sorted(feedIds(feed)), sorted(replay))
        << "seed " << seed << " useMmr " << useMmr << ": feed set != greedy replay set";
    if (replay.size() >= fx.feedSize) {
        EXPECT_EQ(feed.size(), fx.feedSize)
            << "seed " << seed << " useMmr " << useMmr << ": feed not full despite feasible walk";
    }

    // Ranks are contiguous 0..n-1.
    for (std::size_t i = 0; i < feed.size(); ++i) {
        EXPECT_EQ(feed[i].rank, i) << "seed " << seed << " useMmr " << useMmr;
    }
}

} // namespace

TEST(DiversityPropertyTest, NoFeedViolatesAnyConstraint) {
    for (int seed = 0; seed < kNumSeeds; ++seed) {
        rr::Rng rng(static_cast<uint64_t>(9000 + seed));
        const Fixture fx = makeFixture(rng);
        for (bool useMmr : {false, true}) {
            rr::DiversityConfig cfg = fx.config;
            cfg.useMmr = useMmr;
            rr::DiversityReranker reranker(fx.reels, cfg);
            const std::vector<rr::RankedReel> feed = reranker.rerank(fx.user, fx.pool, fx.feedSize);
            checkInvariants(fx, feed, seed, useMmr);
        }
    }
}

TEST(DiversityPropertyTest, Deterministic) {
    for (int seed = 0; seed < kNumSeeds; ++seed) {
        rr::Rng rng(static_cast<uint64_t>(4200 + seed));
        const Fixture fx = makeFixture(rng);
        for (bool useMmr : {false, true}) {
            rr::DiversityConfig cfg = fx.config;
            cfg.useMmr = useMmr;
            rr::DiversityReranker reranker(fx.reels, cfg);
            const std::vector<rr::RankedReel> a = reranker.rerank(fx.user, fx.pool, fx.feedSize);
            const std::vector<rr::RankedReel> b = reranker.rerank(fx.user, fx.pool, fx.feedSize);
            EXPECT_EQ(feedIds(a), feedIds(b)) << "seed " << seed << " useMmr " << useMmr;
        }
    }
}
