#include "rr/learning/online_user_state_updater.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/simulator.hpp"

// Seed-swept property tests for rr::OnlineUserStateUpdater (house style mirrors
// tests/property/behaviour_statistical_test.cpp). Properties asserted:
//   * all three preference vectors stay unit-length after every apply;
//   * same inputs twice => bit-identical output vectors (determinism, D8);
//   * the full generateDataset -> Simulator::step -> apply loop never mutates HiddenUserState
//     (D11 runtime belt-and-braces check; the compile-time guarantee is the apply() signature).

using namespace rr;

namespace {

SimulationConfig smallConfig() {
    SimulationConfig c;
    c.reels = 500;
    c.users = 20;
    c.creators = 40;
    c.topics = 16;
    c.dimensions = 32;
    return c;
}

// Deep value equality for HiddenUserState (the struct owns heap vectors, so a literal byte compare
// of the object is meaningless after copies; this compares every field's value).
bool sameHidden(const HiddenUserState &a, const HiddenUserState &b) {
    return a.userId == b.userId && a.hiddenPreference == b.hiddenPreference &&
           a.preferredTopics == b.preferredTopics &&
           a.preferenceConcentration == b.preferenceConcentration &&
           a.exploreWillingness == b.exploreWillingness &&
           a.avgSessionLength == b.avgSessionLength && a.likePropensity == b.likePropensity &&
           a.sharePropensity == b.sharePropensity && a.durationTolerance == b.durationTolerance &&
           a.preferenceStability == b.preferenceStability;
}

// Run one deterministic driver pass: cold-start every user, then feed each user a fixed pseudo-
// random schedule of reels through Simulator::step + updater.apply. Returns the final users.
std::vector<User> runDriver(const GeneratedDataset &dsIn, uint64_t seed, int stepsPerUser,
                            const LearningConfig &learningCfg) {
    GeneratedDataset ds = dsIn; // local mutable copy (reels/users mutate during the loop)
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

    OnlineUserStateUpdater updater(ds.reels, learningCfg);
    RankingConfig rankingCfg;
    Simulator sim(BehaviourConfig{}, RewardConfig{}, forkRng(seed, "behaviour"),
                  learningCfg.recentWindow, rankingCfg.trendingHalfLifeSeconds);

    Rng pick(seed ^ 0x9e3779b97f4a7c15ULL);
    for (size_t u = 0; u < ds.users.size(); ++u) {
        User &user = ds.users[u];
        const HiddenUserState &hidden = ds.hiddenStates[u];
        for (int s = 0; s < stepsPerUser; ++s) {
            const uint32_t reelIdx = static_cast<uint32_t>(pick.uniformInt(ds.reels.size()));
            Reel &reel = ds.reels[reelIdx];
            const Creator &creator = ds.creators[reel.creatorId.value];
            StepResult step = sim.step(user, hidden, reel, creator);
            updater.apply(user, reel, step.event);
        }
    }
    return ds.users;
}

} // namespace

// All three preference vectors remain unit-length (within embedding.hpp's 1e-4 tolerance) after
// every single apply, swept over >= 20 seeds and arbitrary interaction sequences.
TEST(LearningProperty, PreferenceVectorsStayUnitLengthAcrossSeeds) {
    for (int k = 0; k < 24; ++k) {
        const uint64_t seed = 4000ULL + static_cast<uint64_t>(k) * 7919ULL;
        GeneratedDataset ds = generateDataset(smallConfig(), seed);
        applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

        LearningConfig learningCfg;
        OnlineUserStateUpdater updater(ds.reels, learningCfg);
        RankingConfig rankingCfg;
        Simulator sim(BehaviourConfig{}, RewardConfig{}, forkRng(seed, "behaviour"),
                      learningCfg.recentWindow, rankingCfg.trendingHalfLifeSeconds);

        Rng pick(seed);
        for (size_t u = 0; u < ds.users.size(); ++u) {
            User &user = ds.users[u];
            const HiddenUserState &hidden = ds.hiddenStates[u];
            for (int s = 0; s < 40; ++s) {
                Reel &reel = ds.reels[static_cast<size_t>(pick.uniformInt(ds.reels.size()))];
                const Creator &creator = ds.creators[reel.creatorId.value];
                StepResult step = sim.step(user, hidden, reel, creator);
                updater.apply(user, reel, step.event);

                ASSERT_TRUE(isValid(user.longTermPreference)) << "seed " << seed << " user " << u;
                ASSERT_TRUE(isValid(user.sessionPreference)) << "seed " << seed << " user " << u;
                ASSERT_TRUE(isValid(user.estimatedPreference)) << "seed " << seed << " user " << u;
                // estimatedPreference is the cached effective preference (const-ref helper).
                ASSERT_EQ(&effectivePreference(user), &user.estimatedPreference);
            }
        }
    }
}

// Determinism (D8): running the identical driver twice produces bit-identical preference vectors.
TEST(LearningProperty, DeterministicOutputVectorsAcrossSeeds) {
    LearningConfig learningCfg;
    for (int k = 0; k < 24; ++k) {
        const uint64_t seed = 9000ULL + static_cast<uint64_t>(k) * 6151ULL;
        GeneratedDataset ds = generateDataset(smallConfig(), seed);

        std::vector<User> a = runDriver(ds, seed, /*stepsPerUser=*/30, learningCfg);
        std::vector<User> b = runDriver(ds, seed, /*stepsPerUser=*/30, learningCfg);

        ASSERT_EQ(a.size(), b.size());
        for (size_t u = 0; u < a.size(); ++u) {
            EXPECT_EQ(a[u].estimatedPreference, b[u].estimatedPreference) << "seed " << seed;
            EXPECT_EQ(a[u].longTermPreference, b[u].longTermPreference) << "seed " << seed;
            EXPECT_EQ(a[u].sessionPreference, b[u].sessionPreference) << "seed " << seed;
        }
    }
}

// D11 runtime belt-and-braces: the whole simulation-plus-learning loop never mutates any
// HiddenUserState. (The compile-time guarantee is that apply() has no HiddenUserState parameter;
// this asserts the property end-to-end over real datasets.)
TEST(LearningProperty, HiddenStateUnchangedByLearningLoop) {
    LearningConfig learningCfg;
    for (int k = 0; k < 22; ++k) {
        const uint64_t seed = 12000ULL + static_cast<uint64_t>(k) * 5381ULL;
        GeneratedDataset ds = generateDataset(smallConfig(), seed);
        const std::vector<HiddenUserState> before = ds.hiddenStates; // snapshot

        applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));
        OnlineUserStateUpdater updater(ds.reels, learningCfg);
        RankingConfig rankingCfg;
        Simulator sim(BehaviourConfig{}, RewardConfig{}, forkRng(seed, "behaviour"),
                      learningCfg.recentWindow, rankingCfg.trendingHalfLifeSeconds);

        Rng pick(seed);
        for (size_t u = 0; u < ds.users.size(); ++u) {
            User &user = ds.users[u];
            const HiddenUserState &hidden = ds.hiddenStates[u];
            for (int s = 0; s < 30; ++s) {
                Reel &reel = ds.reels[static_cast<size_t>(pick.uniformInt(ds.reels.size()))];
                const Creator &creator = ds.creators[reel.creatorId.value];
                StepResult step = sim.step(user, hidden, reel, creator);
                updater.apply(user, reel, step.event);
            }
        }

        ASSERT_EQ(before.size(), ds.hiddenStates.size());
        for (size_t u = 0; u < before.size(); ++u) {
            EXPECT_TRUE(sameHidden(before[u], ds.hiddenStates[u]))
                << "seed " << seed << " user " << u;
        }
    }
}
