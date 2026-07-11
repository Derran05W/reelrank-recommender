#include "rr/learning/reward_model.hpp"

#include "rr/infrastructure/config.hpp"
#include "rr/simulation/behaviour_model.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace rr;

namespace {

// A neutral outcome: every engagement signal off, nothing watched. Its reward is exactly 0, so the
// isolated contribution of any single toggled field equals reward(toggled) - 0 = reward(toggled).
BehaviourOutcome neutralOutcome() {
    BehaviourOutcome o{};
    o.baseAffinity = 0.0f;
    o.behaviourScore = 0.0f;
    o.instantSkip = false;
    o.completed = false;
    o.rewatch = false;
    o.liked = false;
    o.shared = false;
    o.followed = false;
    o.notInterested = false;
    o.watchRatio = 0.0f;
    o.watchSeconds = 0.0f;
    o.primaryType = InteractionType::Impression;
    return o;
}

// Mirror of the implementation's documented log-watch normalization (log1p(s)/log1p(120)).
double expectedNormalizedLogWatch(double watchSeconds) {
    return std::log1p(watchSeconds) / std::log1p(120.0);
}

} // namespace

// The neutral outcome must yield exactly zero reward — the baseline the per-term tests build on.
TEST(RewardModelTest, NeutralOutcomeIsZero) {
    RewardModel model{RewardConfig{}};
    EXPECT_FLOAT_EQ(model.reward(neutralOutcome()), 0.0f);
}

// Each term's isolated contribution: toggle exactly one field of the neutral outcome and assert the
// reward equals that term's weight (times the field value where continuous).
TEST(RewardModelTest, PerTermContributions) {
    const RewardConfig cfg{};
    RewardModel model{cfg};

    // watchRatio term: linear in clamp(watchRatio, 0, 1).
    {
        BehaviourOutcome o = neutralOutcome();
        o.watchRatio = 0.6f;
        EXPECT_FLOAT_EQ(model.reward(o), static_cast<float>(cfg.watchRatioWeight * 0.6));
    }
    // watchSeconds term: weight * normalized log1p.
    {
        BehaviourOutcome o = neutralOutcome();
        o.watchSeconds = 30.0f;
        EXPECT_FLOAT_EQ(model.reward(o), static_cast<float>(cfg.watchSecondsWeight *
                                                            expectedNormalizedLogWatch(30.0)));
    }
    // like / share / follow: their exact weights.
    {
        BehaviourOutcome o = neutralOutcome();
        o.liked = true;
        EXPECT_FLOAT_EQ(model.reward(o), static_cast<float>(cfg.likeWeight));
    }
    {
        BehaviourOutcome o = neutralOutcome();
        o.shared = true;
        EXPECT_FLOAT_EQ(model.reward(o), static_cast<float>(cfg.shareWeight));
    }
    {
        BehaviourOutcome o = neutralOutcome();
        o.followed = true;
        EXPECT_FLOAT_EQ(model.reward(o), static_cast<float>(cfg.followWeight));
    }
    // instantSkip / notInterested: their exact negative penalties.
    {
        BehaviourOutcome o = neutralOutcome();
        o.instantSkip = true;
        EXPECT_FLOAT_EQ(model.reward(o), -static_cast<float>(cfg.instantSkipPenalty));
    }
    {
        BehaviourOutcome o = neutralOutcome();
        o.notInterested = true;
        EXPECT_FLOAT_EQ(model.reward(o), -static_cast<float>(cfg.notInterestedPenalty));
    }
}

// The watchRatio term uses clamp(watchRatio, 0, 1): a rewatch (>1) contributes as if 1.0, and the
// term never goes negative.
TEST(RewardModelTest, WatchRatioClampedIntoUnitInterval) {
    const RewardConfig cfg{};
    RewardModel model{cfg};

    BehaviourOutcome rewatch = neutralOutcome();
    rewatch.watchRatio = 1.8f;                                                        // replay
    EXPECT_FLOAT_EQ(model.reward(rewatch), static_cast<float>(cfg.watchRatioWeight)); // as if 1.0

    BehaviourOutcome negative = neutralOutcome();
    negative.watchRatio = -0.5f; // pathological input clamps to 0
    EXPECT_FLOAT_EQ(model.reward(negative), 0.0f);
}

// Bounds: the most positive and most negative achievable outcomes stay inside [-1, 1] (they
// saturate the clamp), and a large sweep of outcomes never escapes the interval.
TEST(RewardModelTest, RewardStaysWithinBounds) {
    RewardModel model{RewardConfig{}};

    // Maximally positive: full watch, long duration, liked+shared+followed. Raw sum > 1 -> clamp.
    BehaviourOutcome best = neutralOutcome();
    best.watchRatio = 1.2f;
    best.watchSeconds = 144.0f;
    best.completed = true;
    best.liked = true;
    best.shared = true;
    best.followed = true;
    EXPECT_FLOAT_EQ(model.reward(best), 1.0f);

    // Maximally negative: instant skip and not interested together. Raw sum < -1 -> clamp.
    BehaviourOutcome worst = neutralOutcome();
    worst.instantSkip = true;
    worst.notInterested = true;
    EXPECT_FLOAT_EQ(model.reward(worst), -1.0f);

    // Sweep arbitrary combinations; all must stay bounded.
    for (int mask = 0; mask < 64; ++mask) {
        BehaviourOutcome o = neutralOutcome();
        o.liked = mask & 1;
        o.shared = mask & 2;
        o.followed = mask & 4;
        o.instantSkip = mask & 8;
        o.notInterested = mask & 16;
        o.completed = mask & 32;
        o.watchRatio = static_cast<float>(mask) / 10.0f;  // 0..6.3
        o.watchSeconds = static_cast<float>(mask) * 5.0f; // 0..315
        const float r = model.reward(o);
        EXPECT_GE(r, -1.0f);
        EXPECT_LE(r, 1.0f);
    }
}

// Config override: non-default weights change the reward accordingly (weights are genuinely wired
// from RewardConfig, not hard-coded).
TEST(RewardModelTest, ConfigWeightsAreHonoured) {
    BehaviourOutcome o = neutralOutcome();
    o.watchRatio = 0.5f;

    RewardConfig defaults{};
    RewardModel defaultModel{defaults};
    EXPECT_FLOAT_EQ(defaultModel.reward(o), static_cast<float>(defaults.watchRatioWeight * 0.5));

    RewardConfig tweaked{};
    tweaked.watchRatioWeight = defaults.watchRatioWeight * 1.6; // still keeps result inside bounds
    RewardModel tweakedModel{tweaked};
    EXPECT_FLOAT_EQ(tweakedModel.reward(o), static_cast<float>(tweaked.watchRatioWeight * 0.5));
    EXPECT_GT(tweakedModel.reward(o), defaultModel.reward(o));

    // A like-only outcome tracks likeWeight under override too.
    BehaviourOutcome likeOnly = neutralOutcome();
    likeOnly.liked = true;
    RewardConfig likeCfg{};
    likeCfg.likeWeight = 0.42;
    EXPECT_FLOAT_EQ(RewardModel{likeCfg}.reward(likeOnly), 0.42f);
}

// Pure-function determinism: the same outcome yields the same reward every call, and two models
// built from the same config agree.
TEST(RewardModelTest, PureFunctionDeterminism) {
    RewardConfig cfg{};
    RewardModel a{cfg};
    RewardModel b{cfg};

    BehaviourOutcome o = neutralOutcome();
    o.watchRatio = 0.73f;
    o.watchSeconds = 41.0f;
    o.completed = true;
    o.liked = true;

    const float first = a.reward(o);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(a.reward(o), first);
    }
    EXPECT_EQ(b.reward(o), first);
}
