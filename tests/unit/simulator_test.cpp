#include "rr/simulation/simulator.hpp"

#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/learning/reward_model.hpp"
#include "rr/recommendation/popularity_recommender.hpp"
#include "rr/recommendation/scoring.hpp"
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

// The engagement increment this event contributes to the trending accumulator: the exact twin of
// the popularity numerator's per-event bump (+1 completion, +2 like, +4 share). Structural: read
// from the outcome the behaviour model actually produced, never assumed.
double trendingIncrement(const BehaviourOutcome &o) {
    return (o.completed ? 1.0 : 0.0) + (o.liked ? 2.0 : 0.0) + (o.shared ? 4.0 : 0.0);
}

// GOLDEN mirror of Simulator::step's creator-affinity update: the observable-flag gains in the
// exact accumulation order used by the implementation (documented constants in simulator.cpp).
// If any impl gain drifts, the mirror diverges on the step where that flag fires.
float expectedAffinityDelta(const BehaviourOutcome &o) {
    return (o.followed ? 0.25f : 0.0f) + (o.shared ? 0.15f : 0.0f) + (o.liked ? 0.10f : 0.0f) +
           (o.completed ? 0.02f : 0.0f) + (o.notInterested ? -0.20f : 0.0f);
}

} // namespace

// Each step increments reel counters exactly in line with the flags of the outcome it returned:
// impressions always, completions/likes/shares/skips iff the corresponding flag fired.
TEST(SimulatorTest, ReelCountersTrackOutcomeFlags) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/11);
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(11, "behaviour"), /*recentWindow=*/20,
                  /*trendingHalfLifeSeconds=*/21600.0};

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
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(5, "behaviour"), /*recentWindow=*/20,
                  /*trendingHalfLifeSeconds=*/21600.0};

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
    Simulator sim{BehaviourConfig{}, rewardCfg, forkRng(7, "behaviour"), /*recentWindow=*/20,
                  /*trendingHalfLifeSeconds=*/21600.0};
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
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(9, "behaviour"), recentWindow,
                  /*trendingHalfLifeSeconds=*/21600.0};

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
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(3, "behaviour"), /*recentWindow=*/50,
                  /*trendingHalfLifeSeconds=*/21600.0};

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

// --- Phase 6: trending accumulator maintenance (TDD 12.4) -------------------------------------

// On every impression the reel's trending accumulators are decayed forward to the event's
// timestamp, then bumped by exactly the popularity-numerator increment for that event, and
// trendingUpdatedAt is advanced. We reconstruct the expected accumulators step-by-step from the
// OBSERVED outcome flags and OBSERVED timestamps, decaying with rr::trendingDecayFactor
// (scoring.hpp) as an INDEPENDENT oracle for the simulator's inlined decay. A small half-life makes
// the decay across the few-second inter-show gaps clearly < 1, so the decay path is genuinely
// exercised.
TEST(SimulatorTest, TrendingAccumulatorDecaysAndAccumulatesVsOracle) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/21);
    const double halfLife = 30.0; // 30 simulated seconds: meaningful decay across each gap
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(21, "behaviour"), /*recentWindow=*/50,
                  halfLife};

    User &user = ds.users[0];
    const HiddenUserState &hidden = ds.hiddenStates[0];
    Reel &reel = ds.reels[0]; // one reel, shown repeatedly, so the accumulator builds over gaps

    double expEngagement = reel.trendingEngagement;
    double expImpressions = reel.trendingImpressions;
    Timestamp prevUpdatedAt = reel.trendingUpdatedAt;

    for (int i = 0; i < 40; ++i) {
        StepResult r = sim.step(user, hidden, reel, creatorFor(ds, reel));
        const Timestamp t = sim.now(); // the event's timestamp (== reel.trendingUpdatedAt now)

        const double w = trendingDecayFactor(prevUpdatedAt, t, halfLife);  // independent oracle
        EXPECT_LT(w, 1.0) << "gap should decay below 1 at this half-life"; // decay path exercised
        expEngagement = expEngagement * w + trendingIncrement(r.outcome);
        expImpressions = expImpressions * w + 1.0;
        prevUpdatedAt = t;

        EXPECT_NEAR(reel.trendingEngagement, expEngagement, 1e-9);
        EXPECT_NEAR(reel.trendingImpressions, expImpressions, 1e-9);
        EXPECT_EQ(reel.trendingUpdatedAt, t); // advanced to the event's timestamp, last
    }
}

// With a huge half-life the decay factor is ~1, so the trending accumulators degenerate into
// UN-decayed twins of the lifetime popularity counters: trendingEngagement == the popularity
// numerator (completion + 2*like + 4*share) and trendingImpressions == impressionCount. This
// pins the "exact exponentially-decayed twin of the popularity numerator" contract.
TEST(SimulatorTest, TrendingAccumulatorIsUndecayedTwinAtHugeHalfLife) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/23);
    const double halfLife = 1e15; // effectively no decay over the simulated seconds here
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(23, "behaviour"), /*recentWindow=*/50,
                  halfLife};

    User &user = ds.users[0];
    const HiddenUserState &hidden = ds.hiddenStates[0];
    Reel &reel = ds.reels[0]; // only this reel is shown, so lifetime counters == these steps

    for (int i = 0; i < 50; ++i) {
        sim.step(user, hidden, reel, creatorFor(ds, reel));
    }

    EXPECT_NEAR(reel.trendingEngagement, popularityEngagement(reel), 1e-4);
    EXPECT_NEAR(reel.trendingImpressions, static_cast<double>(reel.impressionCount), 1e-6);
    EXPECT_EQ(reel.trendingUpdatedAt, sim.now());
}

// --- Phase 6: creator-affinity estimate (TDD 12.6, D11) ---------------------------------------

// User::creatorAffinity is populated ONLY from observable outcome flags (never HiddenUserState),
// accumulated with the documented per-flag gains, and clamped to [0, 1]. We drive a user who is
// strongly aligned with one creator's reel (so completions/likes fire heavily and the estimate
// climbs to the ceiling), and mirror the exact per-flag accumulation as an independent golden.
// This verifies each flag's gain, the [0, 1] clamp (contract for the ranker + creator source),
// and that only the engaged creator ever appears in the map.
TEST(SimulatorTest, CreatorAffinityAccumulatesFromFlagsAndClampsToUnit) {
    GeneratedDataset ds = generateDataset(smallConfig(), /*seed=*/29);
    Simulator sim{BehaviourConfig{}, RewardConfig{}, forkRng(29, "behaviour"), /*recentWindow=*/50,
                  /*trendingHalfLifeSeconds=*/21600.0};

    // Find a reel and force strong alignment so the behaviour model engages consistently. Aligning
    // the (copied) hidden preference to the reel embedding drives high completion probability; this
    // touches ONLY the hidden state handed to the behaviour model (D11 stays intact).
    Reel &reel = ds.reels[0];
    const CreatorId creatorId = reel.creatorId;
    HiddenUserState hidden = ds.hiddenStates[0];
    hidden.hiddenPreference = reel.embedding; // unit vector; a = dot == 1 => strong engagement
    hidden.durationTolerance = 1.0f;          // remove the duration penalty
    hidden.likePropensity = 0.25f;            // upper end of the valid range
    hidden.sharePropensity = 0.10f;

    User &user = ds.users[0];

    float expected = 0.0f; // golden running estimate for this single creator
    bool anySignal = false;

    for (int i = 0; i < 400; ++i) {
        StepResult r = sim.step(user, hidden, reel, creatorFor(ds, reel));

        const float delta = expectedAffinityDelta(r.outcome);
        if (delta != 0.0f) {
            expected = std::clamp(expected + delta, 0.0f, 1.0f);
            anySignal = true;
        }

        // Only the engaged creator can ever appear (we only ever show its reel).
        EXPECT_LE(user.creatorAffinity.size(), 1u);
        if (anySignal) {
            ASSERT_TRUE(user.creatorAffinity.count(creatorId));
            const float actual = user.creatorAffinity.at(creatorId);
            EXPECT_FLOAT_EQ(actual, expected); // gains + clamp match the golden exactly
            EXPECT_GE(actual, 0.0f);           // contract: clamped to [0, 1]
            EXPECT_LE(actual, 1.0f);
        } else {
            EXPECT_TRUE(user.creatorAffinity.empty()); // no signal yet => map untouched
        }
    }

    ASSERT_TRUE(anySignal) << "aligned user should have produced engagement signals";
    // Enough aligned completions/likes over 400 steps to saturate the estimate at the ceiling,
    // proving the upper clamp is actually reached (not just theoretically applied).
    EXPECT_FLOAT_EQ(user.creatorAffinity.at(creatorId), 1.0f);
}
