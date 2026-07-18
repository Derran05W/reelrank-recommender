// Unit tests for the session-health metric group (Phase 16, V2 TDD §4.9/§6). The reduction math is
// exercised on hand-built SessionRecord values — every derived number (U_s under config lambdas,
// per-minute math, mean/median duration, exit-type shares with RunEnded exclusion, next-session
// linkage, harmful-fatigue mean) is checked by hand — plus determinism and the empty-bucket edges.

#include "rr/evaluation/session_health_metrics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace rr;

namespace {

// Build a SessionRecord with the fields the session-health reduction reads.
SessionRecord mk(uint32_t user, SessionExitType exit, uint32_t impressions, float durationSeconds,
                 float satisfactionSum, float regretSum, float harmfulFatigue,
                 float startingSatisfaction = 0.0f) {
    SessionRecord r;
    r.userId = UserId{user};
    r.exitType = exit;
    r.impressions = impressions;
    r.durationSeconds = durationSeconds;
    r.satisfactionSum = satisfactionSum;
    r.regretSum = regretSum;
    r.harmfulFatigue = harmfulFatigue;
    r.startingSatisfaction = startingSatisfaction;
    return r;
}

// A config with DISTINCTIVE, easy-to-read lambdas so the U_s formula is unambiguous under test
// (defaults are a shipped operating point the tests must not depend on).
SessionDynamicsConfig lambdas() {
    SessionDynamicsConfig c;
    c.regretLambda = 2.0;      // λ1
    c.fatigueLambda = 0.5;     // λ2
    c.failureExitLambda = 3.0; // λ3
    return c;
}

} // namespace

// --- U_s free function: exact formula under config lambdas + the failure-exit indicator ----------
TEST(SessionUtilityTest, FormulaWithConfigLambdas) {
    const SessionDynamicsConfig c = lambdas();

    // Failure exit: the λ3 term applies. U_s = 4.0 − 2.0·1.5 − 0.5·0.6 − 3.0·1 = 4 − 3 − 0.3 − 3.
    const SessionRecord fail = mk(1, SessionExitType::Failure, 5, 300.0f, /*sat=*/4.0f,
                                  /*regret=*/1.5f, /*harmful=*/0.6f);
    EXPECT_NEAR(sessionUtility(fail, c), 4.0 - 3.0 - 0.3 - 3.0, 1e-6);

    // Satisfied exit: same components, but NO λ3 term (not a failure). U_s = 4 − 3 − 0.3.
    const SessionRecord sat = mk(1, SessionExitType::Satisfied, 5, 300.0f, 4.0f, 1.5f, 0.6f);
    EXPECT_NEAR(sessionUtility(sat, c), 4.0 - 3.0 - 0.3, 1e-6);

    // Only the FAILURE exit type triggers λ3; the other three closed types do not.
    for (SessionExitType t : {SessionExitType::Fatigue, SessionExitType::External,
                              SessionExitType::Regret, SessionExitType::RunEnded}) {
        const SessionRecord rec = mk(1, t, 5, 300.0f, 4.0f, 1.5f, 0.6f);
        EXPECT_NEAR(sessionUtility(rec, c), 4.0 - 3.0 - 0.3, 1e-6);
    }
}

// --- per-session-minute helper: exact definition + divide-by-zero guard --------------------------
TEST(SessionHealthMathTest, PerSessionMinuteDefinitionAndZeroGuard) {
    // sum 3.0 over 120 duration-seconds = 2.0 minutes -> 1.5 per minute.
    EXPECT_NEAR(perSessionMinute(3.0, 120.0), 1.5, 1e-12);
    // Negative satisfaction sum -> negative per-minute (net dissatisfaction accrues per minute).
    EXPECT_NEAR(perSessionMinute(-3.0, 120.0), -1.5, 1e-12);
    // No duration -> 0, never NaN/inf.
    EXPECT_EQ(perSessionMinute(5.0, 0.0), 0.0);
}

// --- median helper: odd / even / empty / unsorted ------------------------------------------------
TEST(SessionHealthMathTest, MedianOf) {
    EXPECT_EQ(medianOf({}), 0.0);
    EXPECT_EQ(medianOf({7.0}), 7.0);
    EXPECT_EQ(medianOf({3.0, 1.0, 2.0}), 2.0);      // odd -> middle after sort
    EXPECT_EQ(medianOf({4.0, 1.0, 3.0, 2.0}), 2.5); // even -> mean of two middles
}

// --- Overall means + per-minute over a hand-built bucket of CLOSED sessions
// -----------------------
TEST(SessionHealthTest, OverallMeansAndPerMinute) {
    SessionHealthMetrics shm(/*rounds=*/1, lambdas());
    // Three closed sessions. Durations 60/120/60 s -> sum 240 s (4 min). Impressions 2/4/6.
    // Satisfaction sums 1.0/2.0/0.0 -> total 3.0. Regret sums 0.5/0.5/0.0 -> total 1.0.
    shm.add(0, mk(1, SessionExitType::Satisfied, 2, 60.0f, 1.0f, 0.5f, 0.1f));
    shm.add(0, mk(2, SessionExitType::Fatigue, 4, 120.0f, 2.0f, 0.5f, 0.2f));
    shm.add(0, mk(3, SessionExitType::Failure, 6, 60.0f, 0.0f, 0.0f, 0.3f));

    const SessionHealthReport w = shm.reduce();
    EXPECT_EQ(w.sessions, 3u);
    EXPECT_EQ(w.openSessions, 0u);
    EXPECT_NEAR(w.meanDurationSeconds, 240.0 / 3.0, 1e-9);
    EXPECT_NEAR(w.medianDurationSeconds, 60.0, 1e-9); // {60,60,120} -> 60
    EXPECT_NEAR(w.meanImpressions, 12.0 / 3.0, 1e-9);
    EXPECT_NEAR(w.durationMinutes, 4.0, 1e-9);
    // satisfaction per minute = 3.0 / 4.0 min = 0.75; regret per minute = 1.0 / 4.0 = 0.25.
    EXPECT_NEAR(w.satisfactionPerMinute, 0.75, 1e-9);
    EXPECT_NEAR(w.regretPerMinute, 0.25, 1e-9);
    // mean U_s: session utilities are (1−1−0.05)=−0.05, (2−1−0.10)=0.90, (0−0−0.15−3)=−3.15.
    // (fatigueLambda 0.5 * harmful; failure session pays λ3=3.) mean = (−0.05+0.90−3.15)/3.
    EXPECT_NEAR(w.meanSessionUtility, (-0.05 + 0.90 - 3.15) / 3.0, 1e-6);
    EXPECT_NEAR(w.harmfulFatigueMean, (0.1 + 0.2 + 0.3) / 3.0, 1e-6);
}

// --- Exit-type shares + RunEnded exclusion from every denominator --------------------------------
TEST(SessionHealthTest, ExitShareAndRunEndedExclusion) {
    SessionHealthMetrics shm(/*rounds=*/1, lambdas());
    // 2 failure, 1 satisfied, 1 fatigue, 1 regret => 5 CLOSED; plus 3 RunEnded (open).
    shm.add(0, mk(1, SessionExitType::Failure, 3, 30.0f, -0.5f, 0.9f, 0.5f));
    shm.add(0, mk(2, SessionExitType::Failure, 3, 30.0f, -0.5f, 0.9f, 0.5f));
    shm.add(0, mk(3, SessionExitType::Satisfied, 9, 300.0f, 3.0f, 0.0f, 0.0f));
    shm.add(0, mk(4, SessionExitType::Fatigue, 6, 180.0f, 0.5f, 0.2f, 0.7f));
    shm.add(0, mk(5, SessionExitType::Regret, 4, 60.0f, -0.9f, 1.0f, 0.3f));
    // RunEnded records: counted as open, excluded from every mean/rate. Give them WILD values to
    // prove they never leak into the closed-session aggregates.
    shm.add(0, mk(6, SessionExitType::RunEnded, 999, 99999.0f, 99.0f, 99.0f, 99.0f));
    shm.add(0, mk(7, SessionExitType::RunEnded, 999, 99999.0f, 99.0f, 99.0f, 99.0f));
    shm.add(0, mk(8, SessionExitType::RunEnded, 999, 99999.0f, 99.0f, 99.0f, 99.0f));

    const SessionHealthReport w = shm.reduce();
    EXPECT_EQ(w.sessions, 5u);     // closed only
    EXPECT_EQ(w.openSessions, 3u); // RunEnded
    EXPECT_EQ(w.exits.closed(), 5u);
    EXPECT_EQ(w.exits.total(), 8u);

    // Rates are over CLOSED sessions: 2/5 failure, 1/5 satisfied.
    EXPECT_NEAR(w.earlyFailureExitRate, 2.0 / 5.0, 1e-12);
    EXPECT_NEAR(w.naturalCompletionRate, 1.0 / 5.0, 1e-12);

    // Exit shares over closed types sum to 1; RunEnded reported separately as openShare.
    double shareSum = 0.0;
    for (SessionExitType t :
         {SessionExitType::Failure, SessionExitType::Satisfied, SessionExitType::Fatigue,
          SessionExitType::External, SessionExitType::Regret}) {
        shareSum += w.exitShare(t);
    }
    EXPECT_NEAR(shareSum, 1.0, 1e-12);
    EXPECT_EQ(w.exitShare(SessionExitType::RunEnded), 0.0); // not a closed type
    EXPECT_NEAR(w.openShare(), 3.0 / 8.0, 1e-12);

    // The wild RunEnded values NEVER polluted the closed aggregates: closed durations are
    // {30,30,300,180,60} -> mean 120, and the harmful-fatigue mean is over closed only.
    EXPECT_NEAR(w.meanDurationSeconds, (30.0 + 30.0 + 300.0 + 180.0 + 60.0) / 5.0, 1e-9);
    EXPECT_NEAR(w.harmfulFatigueMean, (0.5 + 0.5 + 0.0 + 0.7 + 0.3) / 5.0, 1e-9);
    EXPECT_LT(w.meanDurationSeconds, 1000.0); // definitely no 99999 leaked in
}

// --- Next-session starting-satisfaction linkage: mean over non-first sessions per user
// ------------
TEST(SessionHealthTest, NextSessionLinkage) {
    SessionHealthMetrics shm(/*rounds=*/3, lambdas());
    // User 1: three sessions across rounds 0,1,2 with startingSatisfaction 0.0, 0.4, 0.6.
    //   -> sessions 2 and 3 are "next" sessions (they follow a prior): contribute 0.4 and 0.6.
    shm.add(0, mk(1, SessionExitType::Satisfied, 3, 60.0f, 1.0f, 0.0f, 0.0f, /*start=*/0.0f));
    shm.add(1, mk(1, SessionExitType::Fatigue, 3, 60.0f, 0.5f, 0.1f, 0.2f, /*start=*/0.4f));
    shm.add(2, mk(1, SessionExitType::Regret, 3, 60.0f, -0.2f, 0.5f, 0.3f, /*start=*/0.6f));
    // User 2: a single session -> it is the user's FIRST, so it never contributes to the measure.
    shm.add(0, mk(2, SessionExitType::Satisfied, 3, 60.0f, 1.0f, 0.0f, 0.0f, /*start=*/0.9f));

    const SessionHealthReport w = shm.reduce();
    // Two linked ("next") sessions overall: 0.4 and 0.6 -> mean 0.5.
    EXPECT_EQ(w.linkedSessions, 2u);
    EXPECT_NEAR(w.nextSessionStartingSatisfaction, (0.4 + 0.6) / 2.0, 1e-6);

    // Per-round: round 0 has only first sessions (0 linked); rounds 1 and 2 each have one linked.
    ASSERT_EQ(w.byRound.size(), 3u);
    EXPECT_EQ(w.byRound[0].linkedSessions, 0u);
    EXPECT_EQ(w.byRound[1].linkedSessions, 1u);
    EXPECT_NEAR(w.byRound[1].nextSessionStartingSatisfaction, 0.4, 1e-6);
    EXPECT_EQ(w.byRound[2].linkedSessions, 1u);
    EXPECT_NEAR(w.byRound[2].nextSessionStartingSatisfaction, 0.6, 1e-6);
}

// --- Per-round breakdown: each round's aggregates are independent
// ---------------------------------
TEST(SessionHealthTest, PerRoundBreakdown) {
    SessionHealthMetrics shm(/*rounds=*/2, lambdas());
    shm.add(0, mk(1, SessionExitType::Satisfied, 4, 60.0f, 1.0f, 0.0f, 0.1f));
    shm.add(1, mk(2, SessionExitType::Failure, 2, 30.0f, -0.5f, 0.8f, 0.6f));
    shm.add(1, mk(3, SessionExitType::Fatigue, 6, 90.0f, 0.5f, 0.2f, 0.4f));

    const SessionHealthReport w = shm.reduce();
    ASSERT_EQ(w.byRound.size(), 2u);

    EXPECT_EQ(w.byRound[0].round, 0u);
    EXPECT_EQ(w.byRound[0].sessions, 1u);
    EXPECT_NEAR(w.byRound[0].naturalCompletionRate, 1.0, 1e-12); // the one session is Satisfied
    EXPECT_NEAR(w.byRound[0].earlyFailureExitRate, 0.0, 1e-12);

    EXPECT_EQ(w.byRound[1].round, 1u);
    EXPECT_EQ(w.byRound[1].sessions, 2u);
    EXPECT_NEAR(w.byRound[1].earlyFailureExitRate, 0.5, 1e-12);  // one of two is Failure
    EXPECT_NEAR(w.byRound[1].meanImpressions, 4.0, 1e-9);        // (2+6)/2
    EXPECT_NEAR(w.byRound[1].medianDurationSeconds, 60.0, 1e-9); // {30,90} -> 60
}

// --- Empty accumulator: all zero, no NaN, correctly-sized byRound
// ---------------------------------
TEST(SessionHealthTest, EmptyIsAllZeroAndShaped) {
    SessionHealthMetrics shm(/*rounds=*/3, lambdas());
    const SessionHealthReport w = shm.reduce();
    EXPECT_EQ(w.sessions, 0u);
    EXPECT_EQ(w.openSessions, 0u);
    EXPECT_EQ(w.meanDurationSeconds, 0.0);
    EXPECT_EQ(w.medianDurationSeconds, 0.0);
    EXPECT_EQ(w.satisfactionPerMinute, 0.0);
    EXPECT_EQ(w.regretPerMinute, 0.0);
    EXPECT_EQ(w.meanSessionUtility, 0.0);
    EXPECT_EQ(w.earlyFailureExitRate, 0.0);
    EXPECT_EQ(w.naturalCompletionRate, 0.0);
    EXPECT_EQ(w.harmfulFatigueMean, 0.0);
    EXPECT_EQ(w.nextSessionStartingSatisfaction, 0.0);
    EXPECT_EQ(w.linkedSessions, 0u);
    EXPECT_FALSE(std::isnan(w.satisfactionPerMinute));
    ASSERT_EQ(w.byRound.size(), 3u);
    for (std::size_t r = 0; r < w.byRound.size(); ++r) {
        EXPECT_EQ(w.byRound[r].round, r);
        EXPECT_EQ(w.byRound[r].sessions, 0u);
        EXPECT_EQ(w.byRound[r].satisfactionPerMinute, 0.0);
        EXPECT_EQ(w.byRound[r].medianDurationSeconds, 0.0);
    }
}

// --- Determinism: identical add sequences produce identical reports
// -------------------------------
TEST(SessionHealthTest, DeterministicReduction) {
    const auto build = [] {
        SessionHealthMetrics shm(2, lambdas());
        shm.add(0, mk(1, SessionExitType::Satisfied, 4, 61.0f, 1.0f, 0.2f, 0.1f, 0.0f));
        shm.add(0, mk(2, SessionExitType::Failure, 2, 33.0f, -0.4f, 0.7f, 0.5f, 0.3f));
        shm.add(1, mk(1, SessionExitType::Regret, 3, 47.0f, -0.1f, 0.6f, 0.4f, 0.2f));
        return shm.reduce();
    };
    const SessionHealthReport a = build();
    const SessionHealthReport b = build();
    EXPECT_EQ(a.sessions, b.sessions);
    EXPECT_EQ(a.meanDurationSeconds, b.meanDurationSeconds);
    EXPECT_EQ(a.medianDurationSeconds, b.medianDurationSeconds);
    EXPECT_EQ(a.satisfactionPerMinute, b.satisfactionPerMinute);
    EXPECT_EQ(a.regretPerMinute, b.regretPerMinute);
    EXPECT_EQ(a.meanSessionUtility, b.meanSessionUtility);
    EXPECT_EQ(a.nextSessionStartingSatisfaction, b.nextSessionStartingSatisfaction);
    ASSERT_EQ(a.byRound.size(), b.byRound.size());
    for (std::size_t r = 0; r < a.byRound.size(); ++r) {
        EXPECT_EQ(a.byRound[r].meanSessionUtility, b.byRound[r].meanSessionUtility);
        EXPECT_EQ(a.byRound[r].satisfactionPerMinute, b.byRound[r].satisfactionPerMinute);
    }
}
