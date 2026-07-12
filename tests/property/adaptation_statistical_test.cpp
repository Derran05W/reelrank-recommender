#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>

#include "rr/infrastructure/config.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

// -------------------------------------------------------------------------------------------------
// EXPECTED-FAIL PROTOCOL: this whole test asserts real drift *adaptation dynamics* (reward drops
// then recovers; session-aware recovers faster). Package B builds against the FROZEN DriftScheduler
// interface; the current stub applies no drift, so there is no drop and every assertion here fails.
// The test is written to the frozen contract and the integrator verifies + recalibrates margins
// against the real scheduler at merge.
//
// PROVISIONAL MARGINS (documented, recalibrated by the integrator against real drift). Sourced from
// the Phase 7 published curve magnitudes (results/published/phase7/comparison.md): the hnsw_ranker
// learning arm rides ~0.10 -> ~0.28 reward/impression while the frozen (misaligned-estimate) arm
// sits at ~0.07 and decays. A drift to a topic mix DISJOINT from users' initial preferences makes
// every user's learned estimate momentarily wrong, so the drifted reward should fall toward that
// frozen-arm floor (a drop well over 0.01) and then climb back as learning re-aligns the estimate.
// Margins are intentionally CONSERVATIVE so the qualitative claims hold without overfitting.
constexpr double kDropMargin = 0.01;      // trough is >= this far below the pre-drift baseline
constexpr double kRecoveryMargin = 0.005; // late-window mean is >= this above the trough

// ~300 users / 3000 reels / 150 creators / 16 topics / dim 32 / 80 interactions / feed 10.
ExperimentConfig statConfig() {
    ExperimentConfig c;
    c.simulation.seed = 4242;
    c.simulation.users = 300;
    c.simulation.reels = 3000;
    c.simulation.creators = 150;
    c.simulation.topics = 16;
    c.simulation.dimensions = 32;
    c.simulation.interactionsPerUser = 80; // feed 10 -> 8 rounds
    c.recommendation.feedSize = 10;
    c.recommendation.vectorCandidates = 200;
    c.evaluation.oracleSampleRate = 0.1;
    c.algorithm = RecommendationAlgorithm::HnswRanker;

    // Whole-population drift at interaction 40 (driftRound = 4) to a mix of three HIGH-index topics
    // disjoint from most users' initial 2-5 preferred topics.
    DriftEvent e;
    e.atInteraction = 40;
    e.cohortLo = 0.0;
    e.cohortHi = 1.0;
    e.topicMix = {DriftTopicWeight{13, 1.0}, DriftTopicWeight{14, 1.0}, DriftTopicWeight{15, 1.0}};
    c.drift.events = {e};
    return c;
}

// Mean of the drifted cohort's per-round mean reward over inclusive round range [lo, hi].
double meanDriftedReward(const ExperimentResult &r, long lo, long hi) {
    double sum = 0.0;
    long n = 0;
    const long nRounds = static_cast<long>(r.rounds.size());
    for (long round = std::max<long>(0, lo); round <= hi && round < nRounds; ++round) {
        if (r.rounds[round].driftedImpressions > 0) {
            sum += r.rounds[round].driftedMeanReward;
            ++n;
        }
    }
    return n > 0 ? sum / static_cast<double>(n) : 0.0;
}

// Interactions after drift until the drifted per-round reward first reaches `target` (a COMMON
// absolute level, unlike AdaptationReport::recoveryInteractions which is relative to each arm's
// OWN pre-drift baseline); -1 if never reached.
long interactionsToTarget(const ExperimentResult &r, long driftRound, double target) {
    const long nRounds = static_cast<long>(r.rounds.size());
    const long feedSize = static_cast<long>(r.config.recommendation.feedSize);
    for (long round = driftRound; round < nRounds; ++round) {
        if (r.rounds[round].driftedImpressions > 0 && r.rounds[round].driftedMeanReward >= target) {
            return (round - driftRound + 1) * feedSize;
        }
    }
    return -1;
}

} // namespace

// Two arms, learning ENABLED, same seed: session-aware blend (0.65 / 0.35) vs long-term-only
// (1.0 / 0.0). Asserts (1) the drifted reward drops after drift, (2) it recovers with learning, and
// (3) the session-aware blend adapts FASTER than long-term-only. [EXPECTED-FAIL until sibling
// drift_scheduler.cpp lands.]
TEST(AdaptationStatisticalTest, SessionAwareAdaptsFasterThanLongTermOnly) {
    const fs::path rootSession = fs::path(::testing::TempDir()) / "rr_adapt_session";
    const fs::path rootLongTerm = fs::path(::testing::TempDir()) / "rr_adapt_longterm";
    fs::remove_all(rootSession);
    fs::remove_all(rootLongTerm);

    ExperimentConfig session = statConfig();
    session.learning.enabled = true;
    session.learning.longTermWeight = 0.65;
    session.learning.sessionWeight = 0.35;

    ExperimentConfig longTerm = statConfig();
    longTerm.learning.enabled = true;
    longTerm.learning.longTermWeight = 1.0;
    longTerm.learning.sessionWeight = 0.0;

    const ExperimentResult rs = ExperimentRunner(session, rootSession).run();
    const ExperimentResult rl = ExperimentRunner(longTerm, rootLongTerm).run();

    ASSERT_TRUE(rs.adaptation.configured);
    ASSERT_TRUE(rl.adaptation.configured);
    // Whole-population drift: every user is in the drifted cohort.
    EXPECT_EQ(rs.adaptation.driftedUsers, session.simulation.users);
    EXPECT_EQ(rs.adaptation.controlUsers, 0u);

    const long driftRound = rs.adaptation.driftRound;
    ASSERT_EQ(driftRound, 4);
    const long lastRound = static_cast<long>(rs.rounds.size()) - 1;

    // (1) Reward drops after drift (trough sits below the pre-drift baseline by the drop margin).
    ASSERT_GE(rs.adaptation.preDriftReward, 0.0) << "pre-drift baseline must be defined";
    ASSERT_GE(rs.adaptation.troughRound, driftRound) << "a post-drift trough must exist";
    EXPECT_GT(rs.adaptation.preDriftReward - rs.adaptation.troughReward, kDropMargin)
        << "drifted reward should drop after drift (session arm)";

    // (2) Recovers with learning: the late window (last two rounds) sits above the trough.
    const double lateMean = meanDriftedReward(rs, lastRound - 1, lastRound);
    EXPECT_GT(lateMean, rs.adaptation.troughReward + kRecoveryMargin)
        << "drifted reward should recover after the trough (session arm)";

    // (3) Session-aware adapts faster: higher mean drifted reward over the early adaptation window
    // [driftRound, driftRound+4], and (if both recovered) no later recovery in interactions.
    const double sessionEarly = meanDriftedReward(rs, driftRound, driftRound + 4);
    const double longTermEarly = meanDriftedReward(rl, driftRound, driftRound + 4);
    EXPECT_GT(sessionEarly, longTermEarly)
        << "session-aware blend should recover faster over the early window";

    // NOT compared: AdaptationReport::recoveryInteractions across arms. That field is relative to
    // each arm's OWN pre-drift baseline, and the arms' baselines differ structurally (the
    // long-term-only arm barely personalizes by interaction 40 at eta=0.02, so its 95% bar is far
    // lower and it "recovers" instantly to a worse service level — integration-verified: session
    // 0.2537 baseline / LT-only 0.1755). "Adapts faster" is instead asserted at a COMMON absolute
    // target: 95% of the BETTER arm's pre-drift reward, reached by session-aware no later than by
    // long-term-only.
    const double commonTarget =
        0.95 * std::max(rs.adaptation.preDriftReward, rl.adaptation.preDriftReward);
    const long sessionToTarget = interactionsToTarget(rs, driftRound, commonTarget);
    const long longTermToTarget = interactionsToTarget(rl, driftRound, commonTarget);
    ASSERT_GE(sessionToTarget, 0) << "session-aware must reach the common recovery target";
    if (longTermToTarget >= 0) {
        EXPECT_LE(sessionToTarget, longTermToTarget)
            << "session-aware should reach the common recovery target no later than long-term-only";
    }
}
