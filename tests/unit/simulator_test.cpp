#include "rr/simulation/simulator.hpp"

#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/learning/reward_model.hpp"
#include "rr/simulation/dataset_generator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace rr;

// These tests are STRUCTURAL: they hold for ANY conforming BehaviourModel implementation. They
// never assert a specific behaviour-model output value — only invariants that relate the
// simulator's side effects (reel counters, clock, event fields, user bookkeeping, sessions) to the
// outcome the behaviour model actually returned on that step.

namespace {

SimulationConfig smallConfig() {
    SimulationConfig c;
    c.topics = 8;
    c.creators = 12;
    c.reels = 60;
    c.users = 6;
    c.dimensions = 32;
    return c;
}

// Reel counters must never repeat an item across shows in real feeds, but the simulator does not
// filter — so tests may re-show reels freely; every counter invariant is per-impression.
const Creator &creatorFor(const GeneratedDataset &ds, const Reel &reel) {
    for (const auto &c : ds.creators) {
        if (c.id == reel.creatorId) {
            return c;
        }
    }
    ADD_FAILURE() << "no creator for reel " << reel.id.value;
    return ds.creators.front();
}

Timestamp roundedWatch(float watchSeconds) {
    return static_cast<Timestamp>(std::lround(std::max(0.0f, watchSeconds)));
}

} // namespace

// Each step increments reel counters exactly in line with the flags of the outcome it returned:
// impressions always, completions/likes/shares/skips iff the corresponding flag fired.
TEST(SimulatorTest, ReelCountersTrackOutcomeFlags) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/11);
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(11, "behaviour"), /*recentWindow=*/20};

    for (int round = 0; round < 40; ++round) {
        User &user = ds.users[round % ds.users.size()];
        Reel &reel = ds.reels[(round * 7) % ds.reels.size()];
        const HiddenUserState &hidden = ds.hiddenStates[round % ds.users.size()];

        const uint64_t imp0 = reel.impressionCount;
        const uint64_t comp0 = reel.completionCount;
        const uint64_t like0 = reel.likeCount;
        const uint64_t share0 = reel.shareCount;
        const uint64_t skip0 = reel.skipCount;

        StepResult r = sim.step(user, hidden, reel, creatorFor(ds, reel));

        EXPECT_EQ(reel.impressionCount, imp0 + 1);
        EXPECT_EQ(reel.completionCount, comp0 + (r.outcome.completed ? 1u : 0u));
        EXPECT_EQ(reel.likeCount, like0 + (r.outcome.liked ? 1u : 0u));
        EXPECT_EQ(reel.shareCount, share0 + (r.outcome.shared ? 1u : 0u));
        EXPECT_EQ(reel.skipCount, skip0 + (r.outcome.instantSkip ? 1u : 0u));
    }
}

// The logical clock is monotonically non-decreasing (in fact strictly increasing) and advances each
// step by the rounded watch time plus a FIXED positive browse overhead — never derived from wall
// clock (D9). We infer the overhead from the first step and assert it is constant and positive.
TEST(SimulatorTest, ClockAdvancesByWatchPlusFixedOverhead) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/5);
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(5, "behaviour"), /*recentWindow=*/20};

    EXPECT_EQ(sim.now(), 0u); // starts at 0

    bool haveOverhead = false;
    Timestamp overhead = 0;
    Timestamp prev = sim.now();

    for (int round = 0; round < 40; ++round) {
        User &user = ds.users[round % ds.users.size()];
        Reel &reel = ds.reels[(round * 3) % ds.reels.size()];
        const HiddenUserState &hidden = ds.hiddenStates[round % ds.users.size()];

        StepResult r = sim.step(user, hidden, reel, creatorFor(ds, reel));
        const Timestamp nowAfter = sim.now();

        ASSERT_GE(nowAfter, prev); // non-decreasing
        const Timestamp delta = nowAfter - prev;
        const Timestamp watched = roundedWatch(r.outcome.watchSeconds);
        ASSERT_GE(delta, watched + 1); // strictly advances beyond the watch time
        const Timestamp thisOverhead = delta - watched;
        if (!haveOverhead) {
            overhead = thisOverhead;
            haveOverhead = true;
        }
        EXPECT_EQ(thisOverhead, overhead); // overhead is a fixed constant
        EXPECT_GE(overhead, 1u);           // and positive
        prev = nowAfter;
    }
}

// The assembled event is consistent with the outcome: type is the collapsed primaryType, watch
// fields are copied through, reward equals the RewardModel applied to the outcome, timestamp is the
// clock after the step, and the id fields identify the (user, reel, creator).
TEST(SimulatorTest, EventFieldsConsistentWithOutcome) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/7);
    const RewardConfig rewardCfg{};
    Simulator sim{BehaviourConfig{}, rewardCfg, forkRng(7, "behaviour"), /*recentWindow=*/20};
    RewardModel reference{rewardCfg}; // independent oracle for the reward field

    for (int round = 0; round < 40; ++round) {
        User &user = ds.users[round % ds.users.size()];
        Reel &reel = ds.reels[(round * 5) % ds.reels.size()];
        const HiddenUserState &hidden = ds.hiddenStates[round % ds.users.size()];

        StepResult r = sim.step(user, hidden, reel, creatorFor(ds, reel));
        const InteractionEvent &e = r.event;

        EXPECT_EQ(e.userId, user.id);
        EXPECT_EQ(e.reelId, reel.id);
        EXPECT_EQ(e.creatorId, reel.creatorId);
        EXPECT_EQ(e.type, r.outcome.primaryType);
        EXPECT_EQ(e.watchSeconds, r.outcome.watchSeconds);
        EXPECT_EQ(e.watchRatio, r.outcome.watchRatio);
        EXPECT_EQ(e.reward, reference.reward(r.outcome));
        EXPECT_EQ(e.timestamp, sim.now());
    }
}

// User bookkeeping: the shown reel is marked seen, totalInteractions counts every step, the newest
// event is at the back of recentInteractions, and recentInteractions is bounded by recentWindow.
TEST(SimulatorTest, UserBookkeepingAndRecentWindowBound) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/9);
    const uint32_t recentWindow = 4;
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(9, "behaviour"), recentWindow};

    User &user = ds.users[0];
    const HiddenUserState &hidden = ds.hiddenStates[0];

    const int steps = 12;
    for (int i = 0; i < steps; ++i) {
        Reel &reel = ds.reels[i]; // distinct reels
        const uint64_t total0 = user.totalInteractions;

        StepResult r = sim.step(user, hidden, reel, creatorFor(ds, reel));

        EXPECT_EQ(user.totalInteractions, total0 + 1);
        EXPECT_TRUE(user.seenReels.count(reel.id));
        ASSERT_FALSE(user.recentInteractions.empty());
        // Newest event is at the back and equals the returned event's key fields.
        EXPECT_EQ(user.recentInteractions.back().reelId, r.event.reelId);
        EXPECT_EQ(user.recentInteractions.back().timestamp, r.event.timestamp);
        // Bounded by recentWindow and never exceeds it.
        EXPECT_EQ(user.recentInteractions.size(),
                  std::min<size_t>(static_cast<size_t>(i + 1), recentWindow));
        EXPECT_LE(user.recentInteractions.size(), recentWindow);
    }
    EXPECT_EQ(user.totalInteractions, static_cast<uint64_t>(steps));
    EXPECT_EQ(user.seenReels.size(), static_cast<size_t>(steps));
}

// Sessions rotate: with a short average session length, running many steps for one user advances
// the SessionId at least once, and each rotation resets currentSessionLength (to 1 after the
// interaction that opened the new session). SessionId is monotonic and increments by exactly 1.
TEST(SimulatorTest, SessionRotationAdvancesIdAndResetsLength) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/3);
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(3, "behaviour"), /*recentWindow=*/50};

    User &user = ds.users[0];
    HiddenUserState hidden = ds.hiddenStates[0];
    hidden.avgSessionLength = 3.0f; // force frequent rotations, independent of the behaviour model

    int rotations = 0;
    uint32_t prevSession = 0;
    bool first = true;

    for (int i = 0; i < 80; ++i) {
        Reel &reel = ds.reels[i % ds.reels.size()];
        StepResult r = sim.step(user, hidden, reel, creatorFor(ds, reel));
        const uint32_t session = r.event.sessionId.value;

        if (first) {
            EXPECT_EQ(session, 0u);                   // first session starts at 0
            EXPECT_EQ(user.currentSessionLength, 1u); // opened by this interaction
            first = false;
        } else if (session != prevSession) {
            EXPECT_EQ(session, prevSession + 1);      // rotation increments by exactly 1
            EXPECT_EQ(user.currentSessionLength, 1u); // length reset, then this interaction added
            ++rotations;
        } else {
            EXPECT_GE(user.currentSessionLength, 2u); // same session -> length grew
        }
        EXPECT_GE(session, prevSession); // never decreases
        prevSession = session;
    }

    EXPECT_GT(rotations, 0) << "expected at least one session rotation over 80 short-session steps";
}
