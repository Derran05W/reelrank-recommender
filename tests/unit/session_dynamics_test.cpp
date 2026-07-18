// Phase 16 session-dynamics unit tests (V2 TDD 4.7-4.9). Package A owns the hidden-session
// lifecycle, fatigue dynamics, the probabilistic classified exit model, and the fatigue modulation
// of BehaviourModelV2. The exit/fatigue machinery is exposed as pure free functions
// (rr::startSession / accumulateFatigue / sessionExitProbability / classifyExit / makeSessionRecord
// and rr::fatigueSatisfactionDelta) so each rule is asserted on CONSTRUCTED state with no rng, and
// the Tier-2 "re-entry keeps long-term prefs, fresh session state" acceptance is driven end-to-end
// through Simulator::stepV2 with a forced-exit config (bernoulli(p>=1) is always true).

#include "rr/simulation/simulator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/behaviour_model_v2.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/hidden/hidden_reel_state.hpp"
#include "rr/simulation/hidden/hidden_session_state.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/hidden/latent_reaction.hpp"

using namespace rr;

namespace {

namespace sd = rr::sessionDynamics;

Embedding vec4(float a, float b, float c, float d) { return Embedding{a, b, c, d}; }

LatentReaction latentOf(float sat, float regret = 0.0F, float emo = 0.0F,
                        float fatigueDelta = 0.0F) {
    LatentReaction l{};
    l.immediateSatisfaction = sat;
    l.regret = regret;
    l.emotionalValue = emo;
    l.fatigueDelta = fatigueDelta;
    return l;
}

// Neutral 60s reel on topic 0 / creator 2, no music, no arousal, no novelty. Callers override.
Reel reelOf(TopicId topic = TopicId{0}, CreatorId creator = CreatorId{2}, float duration = 60.0F) {
    Reel r{};
    r.id = ReelId{10};
    r.creatorId = creator;
    r.primaryTopic = topic;
    r.durationSeconds = duration;
    r.emotionalIntensity = 0.0F;
    r.novelty = 0.0F;
    r.musicEmbedding = {};
    return r;
}

} // namespace

// ============================================================================================
//  Fatigue accumulation (V2 TDD 4.7): increments, clamps, EMA, streaks.
// ============================================================================================

TEST(SessionDynamicsTest, TopicAndCreatorFatigueIncrementAndClamp) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    const Reel r = reelOf(TopicId{3}, CreatorId{7});

    accumulateFatigue(cfg, s, r, latentOf(0.0F), 0.0F);
    EXPECT_FLOAT_EQ(s.topicFatigue[TopicId{3}], static_cast<float>(cfg.topicFatigueIncrement));
    EXPECT_FLOAT_EQ(s.creatorFatigue[CreatorId{7}],
                    static_cast<float>(cfg.creatorFatigueIncrement));

    // Repeated exposure keeps rising then saturates at 1.0 (clamped).
    for (int i = 0; i < 50; ++i) {
        accumulateFatigue(cfg, s, r, latentOf(0.0F), 0.0F);
    }
    EXPECT_FLOAT_EQ(s.topicFatigue[TopicId{3}], 1.0F);
    EXPECT_FLOAT_EQ(s.creatorFatigue[CreatorId{7}], 1.0F);
    // A DIFFERENT topic/creator is unaffected (fatigue is per-key).
    EXPECT_EQ(s.topicFatigue.count(TopicId{9}), 0U);
}

TEST(SessionDynamicsTest, FormatFatigueOnRepeatedDurationBucketOnly) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    // First short reel: sets the bucket, no format fatigue yet (nothing to repeat).
    accumulateFatigue(cfg, s, reelOf(TopicId{0}, CreatorId{2}, 6.0F), latentOf(0.0F), 0.0F);
    EXPECT_FLOAT_EQ(s.formatFatigue, 0.0F);
    // Second short reel (same bucket): format fatigue accrues.
    accumulateFatigue(cfg, s, reelOf(TopicId{0}, CreatorId{2}, 8.0F), latentOf(0.0F), 0.0F);
    EXPECT_FLOAT_EQ(s.formatFatigue, static_cast<float>(sd::kFormatFatigueIncrement));
    // A long reel is a DIFFERENT bucket: no further format fatigue.
    accumulateFatigue(cfg, s, reelOf(TopicId{0}, CreatorId{2}, 90.0F), latentOf(0.0F), 0.0F);
    EXPECT_FLOAT_EQ(s.formatFatigue, static_cast<float>(sd::kFormatFatigueIncrement));
}

TEST(SessionDynamicsTest, MusicRepetitionFatigueOnSameMusicCentre) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    Reel a = reelOf();
    a.musicEmbedding = vec4(1, 0, 0, 0);
    Reel same = reelOf();
    same.musicEmbedding = vec4(1, 0, 0, 0); // dot = 1 > threshold
    Reel other = reelOf();
    other.musicEmbedding = vec4(0, 1, 0, 0); // dot = 0 < threshold

    accumulateFatigue(cfg, s, a, latentOf(0.0F), 0.0F); // remembers music
    EXPECT_FLOAT_EQ(s.musicRepetitionFatigue, 0.0F);
    accumulateFatigue(cfg, s, same, latentOf(0.0F), 0.0F); // repeat -> accrue
    EXPECT_FLOAT_EQ(s.musicRepetitionFatigue, static_cast<float>(sd::kMusicRepetitionIncrement));
    accumulateFatigue(cfg, s, other, latentOf(0.0F), 0.0F); // novel music -> no accrual
    EXPECT_FLOAT_EQ(s.musicRepetitionFatigue, static_cast<float>(sd::kMusicRepetitionIncrement));
}

TEST(SessionDynamicsTest, GeneralFatigueTracksLatentFatigueDelta) {
    SessionDynamicsConfig cfg;
    cfg.generalFatigueScale = 1.0;
    HiddenSessionState s;
    accumulateFatigue(cfg, s, reelOf(), latentOf(0.0F, 0.0F, 0.0F, /*fatigueDelta=*/0.3F), 0.0F);
    EXPECT_FLOAT_EQ(s.generalFatigue, 0.3F);
    accumulateFatigue(cfg, s, reelOf(), latentOf(0.0F, 0.0F, 0.0F, 0.3F), 0.0F);
    EXPECT_FLOAT_EQ(s.generalFatigue, 0.6F);
}

TEST(SessionDynamicsTest, EmotionalIntensityFatigueAccumulates) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    Reel intense = reelOf();
    intense.emotionalIntensity = 1.0F;
    accumulateFatigue(cfg, s, intense, latentOf(0.0F), 0.0F);
    EXPECT_FLOAT_EQ(s.emotionalIntensityFatigue,
                    static_cast<float>(sd::kEmotionalIntensityFatigueScale));
}

TEST(SessionDynamicsTest, RemainingAttentionDepletesWithWatchSeconds) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    ASSERT_FLOAT_EQ(s.remainingAttention, 1.0F);
    accumulateFatigue(cfg, s, reelOf(), latentOf(0.0F), /*watchSeconds=*/50.0F);
    EXPECT_FLOAT_EQ(s.remainingAttention,
                    1.0F - static_cast<float>(sd::kAttentionDepletionPerWatchSecond) * 50.0F);
    // Long enough exposure drives it to the [0,1] floor (clamped).
    for (int i = 0; i < 100; ++i) {
        accumulateFatigue(cfg, s, reelOf(), latentOf(0.0F), 50.0F);
    }
    EXPECT_FLOAT_EQ(s.remainingAttention, 0.0F);
}

TEST(SessionDynamicsTest, SatisfactionEmaFollowsTheDocumentedRecurrence) {
    const SessionDynamicsConfig cfg;
    const double a = sd::kSatisfactionEmaAlpha;
    HiddenSessionState s;
    ASSERT_FLOAT_EQ(s.currentSatisfaction, 0.0F);

    accumulateFatigue(cfg, s, reelOf(), latentOf(1.0F), 0.0F);
    EXPECT_NEAR(s.currentSatisfaction, a * 1.0, 1e-5); // (1-a)*0 + a*1

    const double afterFirst = s.currentSatisfaction;
    accumulateFatigue(cfg, s, reelOf(), latentOf(-1.0F), 0.0F);
    EXPECT_NEAR(s.currentSatisfaction, (1.0 - a) * afterFirst + a * -1.0, 1e-5);
}

TEST(SessionDynamicsTest, ConsecutivePoorReelsCountUpThenResetOnAGoodReel) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    accumulateFatigue(cfg, s, reelOf(), latentOf(-0.5F), 0.0F);
    accumulateFatigue(cfg, s, reelOf(), latentOf(-0.2F), 0.0F);
    accumulateFatigue(cfg, s, reelOf(), latentOf(-0.9F), 0.0F);
    EXPECT_EQ(s.consecutivePoorReels, 3U);
    accumulateFatigue(cfg, s, reelOf(), latentOf(0.4F), 0.0F); // above the poor threshold
    EXPECT_EQ(s.consecutivePoorReels, 0U);
    EXPECT_EQ(s.impressionsThisSession, 4U);
}

// ============================================================================================
//  Away-time decay + session start (V2 TDD 4.7): fatigue halves at the half-life, carry-over.
// ============================================================================================

TEST(SessionDynamicsTest, AwayDecayHalvesFatigueAtOneHalfLife) {
    SessionDynamicsConfig cfg;
    cfg.awayDecayHalfLifeSeconds = 3600.0;
    HiddenSessionState s;
    s.generalFatigue = 0.8F;
    s.formatFatigue = 0.6F;
    s.musicRepetitionFatigue = 0.4F;
    s.emotionalIntensityFatigue = 0.5F;
    s.topicFatigue[TopicId{1}] = 0.9F;
    s.creatorFatigue[CreatorId{2}] = 0.7F;
    s.currentSatisfaction = 0.6F;
    s.previousSessionEnd = 1000;
    // A used session that must be scrubbed on re-entry:
    s.impressionsThisSession = 5;
    s.accumulatedRegret = 2.0F;
    s.consecutivePoorReels = 3;

    startSession(cfg, s, /*now=*/1000 + 3600); // exactly one half-life away

    EXPECT_NEAR(s.generalFatigue, 0.4F, 1e-5);
    EXPECT_NEAR(s.formatFatigue, 0.3F, 1e-5);
    EXPECT_NEAR(s.musicRepetitionFatigue, 0.2F, 1e-5);
    EXPECT_NEAR(s.emotionalIntensityFatigue, 0.25F, 1e-5);
    EXPECT_NEAR(s.topicFatigue[TopicId{1}], 0.45F, 1e-5);
    EXPECT_NEAR(s.creatorFatigue[CreatorId{2}], 0.35F, 1e-5);
    // Carried-over satisfaction fades at the same half-life and seeds the new session.
    EXPECT_NEAR(s.startingSatisfaction, 0.3F, 1e-5);
    EXPECT_NEAR(s.currentSatisfaction, 0.3F, 1e-5);
    // Per-session accumulators are fresh.
    EXPECT_EQ(s.impressionsThisSession, 0U);
    EXPECT_FLOAT_EQ(s.accumulatedRegret, 0.0F);
    EXPECT_EQ(s.consecutivePoorReels, 0U);
    EXPECT_FLOAT_EQ(s.remainingAttention, 1.0F);
    EXPECT_EQ(s.sessionStartTime, 1000 + 3600);
}

TEST(SessionDynamicsTest, TwoHalfLivesAwayQuartersFatigue) {
    SessionDynamicsConfig cfg;
    cfg.awayDecayHalfLifeSeconds = 3600.0;
    HiddenSessionState s;
    s.generalFatigue = 0.8F;
    s.previousSessionEnd = 100;
    startSession(cfg, s, 100 + 2 * 3600);
    EXPECT_NEAR(s.generalFatigue, 0.2F, 1e-5);
}

TEST(SessionDynamicsTest, FirstEverSessionAppliesNoDecay) {
    SessionDynamicsConfig cfg;
    HiddenSessionState s;
    s.generalFatigue = 0.8F;  // pre-set to prove the guard
    s.previousSessionEnd = 0; // never had a previous session
    startSession(cfg, s, /*now=*/999999);
    EXPECT_FLOAT_EQ(s.generalFatigue, 0.8F); // awayGap forced 0 -> no decay
}

// ============================================================================================
//  Probabilistic exit logit (V2 TDD 4.8): each coefficient's MONOTONE effect on P(exit).
// ============================================================================================

TEST(SessionDynamicsTest, ExitProbabilityMonotoneInEachDriver) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState base;
    base.impressionsThisSession = 10;
    const double p0 = sessionExitProbability(cfg, base, /*interruption=*/false);

    HiddenSessionState moreFatigue = base;
    moreFatigue.generalFatigue = 0.9F;
    EXPECT_GT(sessionExitProbability(cfg, moreFatigue, false), p0) << "fatigue must raise P(exit)";

    HiddenSessionState moreRegret = base;
    moreRegret.accumulatedRegret = 8.0F; // mean regret 0.8
    EXPECT_GT(sessionExitProbability(cfg, moreRegret, false), p0) << "regret must raise P(exit)";

    HiddenSessionState moreStreak = base;
    moreStreak.consecutivePoorReels = 6;
    EXPECT_GT(sessionExitProbability(cfg, moreStreak, false), p0)
        << "a poor streak must raise P(exit)";

    HiddenSessionState moreSat = base;
    moreSat.currentSatisfaction = 0.9F;
    EXPECT_LT(sessionExitProbability(cfg, moreSat, false), p0) << "satisfaction must LOWER P(exit)";

    EXPECT_GT(sessionExitProbability(cfg, base, /*interruption=*/true), p0)
        << "an external interruption must raise P(exit)";
}

// ============================================================================================
//  Exit classification (V2 TDD 4.8 taxonomy): one test per label, incl. RunEnded exclusion.
// ============================================================================================

TEST(SessionDynamicsTest, ClassifyExternalWhenInterrupted) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s; // even a "satisfied-looking" session -> External wins when interrupted
    s.impressionsThisSession = 30;
    s.currentSatisfaction = 0.8F;
    EXPECT_EQ(classifyExit(cfg, s, /*interruption=*/true), SessionExitType::External);
}

TEST(SessionDynamicsTest, ClassifyFailureOnEarlyPoorSession) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    s.impressionsThisSession = 2;  // < kFailureMaxImpressions
    s.currentSatisfaction = -0.3F; // < kFailureSatisfactionThreshold
    EXPECT_EQ(classifyExit(cfg, s, false), SessionExitType::Failure);
}

TEST(SessionDynamicsTest, ClassifyRegretOnRegretHeavySession) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    s.impressionsThisSession = 10; // not early
    s.currentSatisfaction = 0.2F;  // not poor
    s.accumulatedRegret = 8.0F;    // mean 0.8 > kRegretExitMeanThreshold
    EXPECT_EQ(classifyExit(cfg, s, false), SessionExitType::Regret);
}

TEST(SessionDynamicsTest, ClassifyFatigueOnDepletion) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState highFatigue;
    highFatigue.impressionsThisSession = 12;
    highFatigue.currentSatisfaction = 0.2F;
    highFatigue.generalFatigue = 0.75F; // > kFatigueExitThreshold
    EXPECT_EQ(classifyExit(cfg, highFatigue, false), SessionExitType::Fatigue);

    HiddenSessionState noAttention;
    noAttention.impressionsThisSession = 12;
    noAttention.currentSatisfaction = 0.2F;
    noAttention.remainingAttention = 0.1F; // < kAttentionDepletedThreshold
    EXPECT_EQ(classifyExit(cfg, noAttention, false), SessionExitType::Fatigue);
}

TEST(SessionDynamicsTest, ClassifySatisfiedOnLongProductiveSession) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    s.impressionsThisSession = 25;
    s.currentSatisfaction = 0.5F;
    s.accumulatedRegret = 1.0F; // mean 0.04, low
    s.generalFatigue = 0.3F;
    s.remainingAttention = 0.6F;
    EXPECT_EQ(classifyExit(cfg, s, false), SessionExitType::Satisfied);
}

TEST(SessionDynamicsTest, ClassifyNeverReturnsRunEnded) {
    const SessionDynamicsConfig cfg;
    for (uint32_t imp : {1U, 3U, 6U, 20U}) {
        for (float sat : {-0.9F, -0.1F, 0.0F, 0.4F, 0.9F}) {
            for (float fat : {0.0F, 0.4F, 0.8F}) {
                for (float att : {0.05F, 0.5F, 1.0F}) {
                    for (float reg : {0.0F, 4.0F, 9.0F}) {
                        for (bool intr : {false, true}) {
                            HiddenSessionState s;
                            s.impressionsThisSession = imp;
                            s.currentSatisfaction = sat;
                            s.generalFatigue = fat;
                            s.remainingAttention = att;
                            s.accumulatedRegret = reg;
                            EXPECT_NE(classifyExit(cfg, s, intr), SessionExitType::RunEnded);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================================
//  Session outcome record (V2 TDD 4.9): U_s, harmful fatigue, duration.
// ============================================================================================

TEST(SessionDynamicsTest, SessionRecordComputesUtilityHarmfulFatigueAndDuration) {
    SessionDynamicsConfig cfg;
    cfg.regretLambda = 1.0;
    cfg.fatigueLambda = 0.5;
    cfg.failureExitLambda = 2.0;
    HiddenSessionState s;
    s.satisfactionSum = 6.0F;
    s.regretSum = 2.0F;
    s.generalFatigue = 0.8F; // harmful beyond 0.5 -> 0.3
    s.sessionStartTime = 100;
    s.impressionsThisSession = 12;
    s.startingSatisfaction = 0.15F;

    const SessionRecord satisfied =
        makeSessionRecord(cfg, s, UserId{4}, SessionId{2}, SessionExitType::Satisfied, 400);
    EXPECT_EQ(satisfied.userId.value, 4U);
    EXPECT_EQ(satisfied.sessionId.value, 2U);
    EXPECT_EQ(satisfied.impressions, 12U);
    EXPECT_FLOAT_EQ(satisfied.durationSeconds, 300.0F);
    EXPECT_NEAR(satisfied.harmfulFatigue, 0.3F, 1e-5);
    EXPECT_FLOAT_EQ(satisfied.startingSatisfaction, 0.15F);
    // U_s = 6 - 1*2 - 0.5*0.3 - 2*0 (not a failure) = 3.85
    EXPECT_NEAR(satisfied.sessionUtility, 3.85F, 1e-4);

    // A failure exit takes the extra failureExitLambda penalty.
    const SessionRecord failure =
        makeSessionRecord(cfg, s, UserId{4}, SessionId{2}, SessionExitType::Failure, 400);
    EXPECT_NEAR(failure.sessionUtility, 3.85F - 2.0F, 1e-4);
}

TEST(SessionDynamicsTest, HarmfulFatigueIsZeroBelowThreshold) {
    const SessionDynamicsConfig cfg;
    HiddenSessionState s;
    s.generalFatigue = 0.3F; // below kHarmfulFatigueThreshold
    const SessionRecord rec =
        makeSessionRecord(cfg, s, UserId{1}, SessionId{0}, SessionExitType::Satisfied, 10);
    EXPECT_FLOAT_EQ(rec.harmfulFatigue, 0.0F);
}

// ============================================================================================
//  Fatigue modulation (V2 TDD 4.7): the plan formula + P13 tolerance-trait modulation.
// ============================================================================================

TEST(SessionDynamicsTest, FatigueReducesSatisfactionMonotonically) {
    const SessionDynamicsConfig cfg;
    HiddenUserState u{};
    u.repetitionTolerance = 0.0F;
    u.creatorLoyalty = 0.0F;
    const Reel r = reelOf(TopicId{0}, CreatorId{2});

    HiddenSessionState fresh;
    EXPECT_NEAR(fatigueSatisfactionDelta(cfg, u, r, fresh), 0.0, 1e-9)
        << "no fatigue -> no adjustment";

    HiddenSessionState tired;
    tired.topicFatigue[TopicId{0}] = 0.5F;
    const double d1 = fatigueSatisfactionDelta(cfg, u, r, tired);
    EXPECT_LT(d1, 0.0) << "topic fatigue must lower satisfaction";

    HiddenSessionState moreTired = tired;
    moreTired.topicFatigue[TopicId{0}] = 0.9F;
    EXPECT_LT(fatigueSatisfactionDelta(cfg, u, r, moreTired), d1) << "more fatigue, larger drop";

    HiddenSessionState creatorTired;
    creatorTired.creatorFatigue[CreatorId{2}] = 0.8F;
    EXPECT_LT(fatigueSatisfactionDelta(cfg, u, r, creatorTired), 0.0)
        << "creator fatigue must lower satisfaction";
}

TEST(SessionDynamicsTest, RepetitionToleranceAndCreatorLoyaltyDampenFatiguePenalty) {
    const SessionDynamicsConfig cfg;
    const Reel r = reelOf(TopicId{0}, CreatorId{2});

    HiddenSessionState topicTired;
    topicTired.topicFatigue[TopicId{0}] = 0.8F;
    HiddenUserState intolerant{};
    intolerant.repetitionTolerance = 0.0F;
    HiddenUserState tolerant{};
    tolerant.repetitionTolerance = 0.9F;
    EXPECT_GT(fatigueSatisfactionDelta(cfg, tolerant, r, topicTired),
              fatigueSatisfactionDelta(cfg, intolerant, r, topicTired))
        << "a repetition-tolerant user feels LESS topic-fatigue drop (delta closer to 0)";

    HiddenSessionState creatorTired;
    creatorTired.creatorFatigue[CreatorId{2}] = 0.8F;
    HiddenUserState disloyal{};
    disloyal.creatorLoyalty = 0.0F;
    HiddenUserState loyal{};
    loyal.creatorLoyalty = 0.9F;
    EXPECT_GT(fatigueSatisfactionDelta(cfg, loyal, r, creatorTired),
              fatigueSatisfactionDelta(cfg, disloyal, r, creatorTired))
        << "a creator-loyal user feels LESS creator-fatigue drop";
}

TEST(SessionDynamicsTest, NoveltyNeedMeetingNovelContentAddsUtility) {
    const SessionDynamicsConfig cfg;
    HiddenUserState u{};
    Reel novel = reelOf(TopicId{0}, CreatorId{2});
    novel.novelty = 1.0F;
    Reel stale = reelOf(TopicId{0}, CreatorId{2});
    stale.novelty = 0.0F;

    HiddenSessionState hungry;
    hungry.noveltyNeed = 1.0F;
    EXPECT_GT(fatigueSatisfactionDelta(cfg, u, novel, hungry),
              fatigueSatisfactionDelta(cfg, u, stale, hungry))
        << "novel content satisfies novelty need (gamma * noveltyNeed * novelty > 0)";
}

// simulate() applies the modulation to BOTH latentOut (welfare truth) and the sampled observables.
TEST(SessionDynamicsTest, SimulateFatigueLowersLatentTruthAndWatchBehaviour) {
    BehaviourModelV2 model(BehaviourConfig{}, BehaviourV2Config{});
    model.configureSessionDynamics(SessionDynamicsConfig{});

    HiddenUserState u{};
    u.hiddenPreference = vec4(1, 0, 0, 0);
    u.repetitionTolerance = 0.0F;
    Reel r{};
    r.id = ReelId{1};
    r.creatorId = CreatorId{0};
    r.embedding = vec4(1, 0, 0, 0); // aligned -> positive base satisfaction
    r.primaryTopic = TopicId{0};
    r.durationSeconds = 30.0F;
    HiddenReelState hr{};
    hr.reelId = ReelId{1};
    Creator c{};
    c.id = CreatorId{0};
    c.styleEmbedding = vec4(1, 0, 0, 0);

    HiddenSessionState tired;
    tired.topicFatigue[TopicId{0}] = 1.0F;
    tired.generalFatigue = 1.0F;

    // Average the latent + watch over many draws with vs without the fatigued session.
    const int n = 4000;
    double satFresh = 0, satTired = 0, watchFresh = 0, watchTired = 0;
    Rng bF(1), sF(2), bT(1), sT(2);
    for (int i = 0; i < n; ++i) {
        LatentReaction lf, lt;
        const BehaviourOutcome of = model.simulate(u, r, hr, c, bF, sF, lf, /*session=*/nullptr);
        const BehaviourOutcome ot = model.simulate(u, r, hr, c, bT, sT, lt, &tired);
        satFresh += lf.immediateSatisfaction;
        satTired += lt.immediateSatisfaction;
        watchFresh += of.watchRatio;
        watchTired += ot.watchRatio;
    }
    EXPECT_LT(satTired / n, satFresh / n - 0.1)
        << "fatigue must lower the welfare-visible latent satisfaction (fresh=" << satFresh / n
        << " tired=" << satTired / n << ")";
    EXPECT_LT(watchTired / n, watchFresh / n)
        << "fatigue must lower observed watch too (behaviour reflects the adjusted latent)";
}

// ============================================================================================
//  Tier-2 acceptance: re-entry keeps long-term prefs, gets a FRESH session state.
//  Driven through Simulator::stepV2 with a forced-exit config (every impression closes its
//  session), so each impression is a fresh single-impression session.
// ============================================================================================

TEST(SessionDynamicsTest, ReEntryPreservesLongTermPrefsWithFreshSessionState) {
    SimulationConfig sc;
    sc.users = 4;
    sc.reels = 60;
    sc.creators = 8;
    sc.topics = 8;
    sc.dimensions = 16;
    RealismConfig realism;
    realism.contentV2 = true;
    realism.latentReactions = true;
    realism.sessionDynamics = true;
    const uint64_t seed = 20260716;
    GeneratedDataset ds = generateDataset(sc, realism, seed);

    // Force an exit on EVERY impression: exitBias huge positive -> sigmoid ~ 1 -> bernoulli always
    // fires. externalInterruptionHazard 0 so exits classify by state, not interruption.
    SessionDynamicsConfig sdc;
    sdc.exitBias = 60.0;
    sdc.externalInterruptionHazard = 0.0;

    Simulator sim(BehaviourConfig{}, BehaviourV2Config{}, sdc, RewardConfig{},
                  forkRng(seed, "behaviour"), forkRng(seed, "satisfaction"),
                  forkRng(seed, "session-exit"), forkRng(seed, "external-interruption"),
                  /*recentWindow=*/20, /*trendingHalfLifeSeconds=*/3600.0);

    const HiddenUserState prefsBefore = ds.hiddenStates[0]; // snapshot the long-term (hidden) prefs
    User &user = ds.users[0];

    std::vector<SessionRecord> records;
    for (int i = 0; i < 6; ++i) {
        Reel &reel = ds.reels[(i * 7) % ds.reels.size()];
        StepV2Inputs v2;
        v2.hiddenReel = &ds.hiddenReelStates[(i * 7) % ds.reels.size()];
        v2.requestId = static_cast<uint64_t>(i + 1);
        v2.requestTimestamp = sim.now();
        LatentReaction latent;
        SessionRecord closed{};
        const StepResult sr = sim.stepV2(user, ds.hiddenStates[0], reel,
                                         ds.creators[reel.creatorId.value], v2, latent, &closed);
        ASSERT_TRUE(sr.event.observedExitAfterImpression)
            << "forced exit must fire every impression";
        records.push_back(closed);
        // The V1-visible session length resets to 0 after each exit (coherent with V1 bookkeeping).
        EXPECT_EQ(user.currentSessionLength, 0U);
    }

    // Long-term prefs (hidden) are structurally const in stepV2 — assert the key channels/traits
    // are byte-identical after six sessions (the Tier-2 "preserve long-term preferences" clause).
    EXPECT_EQ(ds.hiddenStates[0].hiddenPreference, prefsBefore.hiddenPreference);
    EXPECT_EQ(ds.hiddenStates[0].preferredTopics, prefsBefore.preferredTopics);
    EXPECT_FLOAT_EQ(ds.hiddenStates[0].avgSessionLength, prefsBefore.avgSessionLength);
    EXPECT_FLOAT_EQ(ds.hiddenStates[0].repetitionTolerance, prefsBefore.repetitionTolerance);
    EXPECT_FLOAT_EQ(ds.hiddenStates[0].creatorLoyalty, prefsBefore.creatorLoyalty);

    // Each closed session is FRESH: exactly the one impression it contained (state reset every
    // time), and the V1-visible session id advances monotonically.
    for (const SessionRecord &rec : records) {
        EXPECT_EQ(rec.impressions, 1U)
            << "session state must reset -> impressions never accumulate";
        EXPECT_NE(rec.exitType, SessionExitType::RunEnded);
    }
    for (size_t i = 1; i < records.size(); ++i) {
        EXPECT_EQ(records[i].sessionId.value, records[i - 1].sessionId.value + 1)
            << "the V1-visible session id rotates once per exit";
    }
    // Starting satisfaction is CARRIED (decayed) across sessions: the very first session starts
    // neutral, later sessions inherit the previous session's fading mood (non-zero in general).
    EXPECT_FLOAT_EQ(records[0].startingSatisfaction, 0.0F);
}
