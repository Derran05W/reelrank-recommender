#include "rr/simulation/user_generator.hpp"

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

using namespace rr;

namespace {

// Self-contained topic factory (does not depend on the parallel TopicGenerator package): N
// normalized random topic centres. Uses only rr::Rng + rr::normalize.
std::vector<Topic> makeTopics(uint32_t count, uint32_t dim, uint64_t seed) {
    Rng rng(seed);
    std::vector<Topic> topics;
    topics.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Embedding c(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            c[d] = static_cast<float>(rng.gaussian());
        }
        normalize(c);
        topics.push_back(Topic{TopicId{i}, std::move(c)});
    }
    return topics;
}

// A trait is "in range" if it lands within [lo, hi] allowing a small float-rounding tolerance
// (rng.uniform draws a double in [lo, hi); casting to float can nudge it a hair past a bound).
constexpr double kTol = 1e-4;
::testing::AssertionResult inRange(const char *name, double v, double lo, double hi) {
    if (v >= lo - kTol && v <= hi + kTol) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << name << " = " << v << " outside [" << lo << ", " << hi << "]";
}

SimulationConfig cfg(uint32_t users, uint32_t dim) {
    SimulationConfig c;
    c.users = users;
    c.dimensions = dim;
    return c;
}

} // namespace

TEST(UserGeneratorTest, HiddenPreferenceValidAndNormalized) {
    const uint32_t dim = 64;
    auto topics = makeTopics(16, dim, 7);
    Rng rng(42);
    auto gen = generateUsers(cfg(300, dim), topics, rng);

    ASSERT_EQ(gen.hiddenStates.size(), 300u);
    for (const auto &h : gen.hiddenStates) {
        EXPECT_EQ(h.hiddenPreference.size(), dim);
        EXPECT_TRUE(isValid(h.hiddenPreference)) << "user " << h.userId.value;
    }
}

TEST(UserGeneratorTest, PreferredTopicCountInRange) {
    const uint32_t dim = 64;
    auto topics = makeTopics(16, dim, 7); // >= kMaxPreferredTopics, so no clamping
    Rng rng(42);
    auto gen = generateUsers(cfg(300, dim), topics, rng);

    for (const auto &h : gen.hiddenStates) {
        EXPECT_GE(h.preferredTopics.size(), 2u);
        EXPECT_LE(h.preferredTopics.size(), 5u);
        // Distinct and in-bounds.
        std::set<uint32_t> seen;
        for (TopicId t : h.preferredTopics) {
            EXPECT_LT(t.value, 16u);
            EXPECT_TRUE(seen.insert(t.value).second) << "duplicate topic " << t.value;
        }
    }
}

TEST(UserGeneratorTest, BehaviouralTraitsWithinRange) {
    const uint32_t dim = 64;
    auto topics = makeTopics(16, dim, 7);
    Rng rng(123);
    auto gen = generateUsers(cfg(500, dim), topics, rng);

    for (const auto &h : gen.hiddenStates) {
        using namespace userTraits;
        EXPECT_TRUE(inRange("concentration", h.preferenceConcentration, kConcentrationLo,
                            kConcentrationHi));
        EXPECT_TRUE(inRange("explore", h.exploreWillingness, kExploreLo, kExploreHi));
        EXPECT_TRUE(inRange("sessionLen", h.avgSessionLength, kSessionLengthLo, kSessionLengthHi));
        EXPECT_TRUE(inRange("like", h.likePropensity, kLikePropensityLo, kLikePropensityHi));
        EXPECT_TRUE(inRange("share", h.sharePropensity, kSharePropensityLo, kSharePropensityHi));
        EXPECT_TRUE(
            inRange("duration", h.durationTolerance, kDurationToleranceLo, kDurationToleranceHi));
        EXPECT_TRUE(inRange("stability", h.preferenceStability, kStabilityLo, kStabilityHi));
    }
}

TEST(UserGeneratorTest, IdsAlignedAcrossVectors) {
    const uint32_t dim = 32;
    auto topics = makeTopics(10, dim, 3);
    Rng rng(9);
    const uint32_t n = 250;
    auto gen = generateUsers(cfg(n, dim), topics, rng);

    ASSERT_EQ(gen.users.size(), n);
    ASSERT_EQ(gen.hiddenStates.size(), n);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(gen.users[i].id, UserId{i});
        EXPECT_EQ(gen.hiddenStates[i].userId, UserId{i});
        EXPECT_EQ(gen.users[i].id, gen.hiddenStates[i].userId);
    }
}

// Public User carries no hidden state (D11) and no estimated-preference state yet (Phase 4 does
// cold start): estimated/long-term/session preferences are empty and all counters/collections zero.
TEST(UserGeneratorTest, PublicUserHasNoEstimatedState) {
    const uint32_t dim = 64;
    auto topics = makeTopics(16, dim, 7);
    Rng rng(42);
    auto gen = generateUsers(cfg(50, dim), topics, rng);

    for (const auto &u : gen.users) {
        EXPECT_TRUE(u.estimatedPreference.empty());
        EXPECT_TRUE(u.longTermPreference.empty());
        EXPECT_TRUE(u.sessionPreference.empty());
        EXPECT_TRUE(u.seenReels.empty());
        EXPECT_TRUE(u.creatorAffinity.empty());
        EXPECT_TRUE(u.recentInteractions.empty());
        EXPECT_EQ(u.totalInteractions, 0u);
        EXPECT_EQ(u.currentSessionLength, 0u);
    }
}

// Generating zero users is well-defined: both output vectors are empty, no crash, no throw (even
// with empty topics, since no preferences are built).
TEST(UserGeneratorTest, ZeroUsersDoesNotCrash) {
    const uint32_t dim = 64;
    auto topics = makeTopics(8, dim, 1);
    Rng rng(42);
    auto gen = generateUsers(cfg(0, dim), topics, rng);
    EXPECT_TRUE(gen.users.empty());
    EXPECT_TRUE(gen.hiddenStates.empty());

    std::vector<Topic> none;
    Rng rng2(42);
    auto gen2 = generateUsers(cfg(0, dim), none, rng2);
    EXPECT_TRUE(gen2.users.empty());
    EXPECT_TRUE(gen2.hiddenStates.empty());
}

// Empty topics with users > 0 is a setup error (D10): fail fast.
TEST(UserGeneratorTest, EmptyTopicsWithUsersThrows) {
    std::vector<Topic> none;
    Rng rng(42);
    EXPECT_THROW(generateUsers(cfg(10, 64), none, rng), std::invalid_argument);
}
