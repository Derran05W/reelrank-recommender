// Event-driven runner PIPELINE integration test (Phase 18, V2 TDD §4.11/4.12/4.14, D20/D22).
// Drives a tiny full-gate event_queue experiment end-to-end through the ExperimentRunner (which
// dispatches to the EventDrivenRunner) and asserts the Phase-18 exit criteria that need the whole
// pipeline: all four §6 metric groups plus the event-mode session-health additions are emitted;
// the same seed reproduces an identical event-log digest AND byte-identical welfare/session CSVs;
// users sit on independent open/exit/return timelines; and the legacy round-robin path on the same
// config (minus the scheduler) is untouched and emits none of the event-mode additions.
//
// Pure pieces (digest fold, return-delay formula, phase ordering) are unit-tested in
// event_driven_runner_test.cpp; the pinned tie-breaker golden values are package C's.

#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace rr;

namespace {

namespace fs = std::filesystem;

// Tiny full-gate event config: 200 users / 2000 reels / horizon = 6 simulated hours. content_v2 +
// latent_reactions + session_dynamics are all on (event_queue requires the full stack). Opens are
// staggered over the first hour and the baseline return delay is ~1h, so users genuinely open,
// scroll, exit, and return at different simulated times within the window.
ExperimentConfig eventConfig(uint64_t seed = 20260718) {
    ExperimentConfig c;
    c.simulation.seed = seed;
    c.simulation.users = 200;
    c.simulation.reels = 2000;
    c.simulation.creators = 40;
    c.simulation.topics = 8;
    c.simulation.dimensions = 16;
    c.simulation.scheduler = "event_queue";
    c.simulation.horizonSeconds = 6.0 * 3600.0; // 6 simulated hours
    c.recommendation.feedSize = 10;
    c.recommendation.vectorCandidates = 100;
    c.evaluation.oracleSampleRate = 0.1;
    c.evaluation.retrievalSampleRate = 0.05;
    c.algorithm = RecommendationAlgorithm::HnswRanker;
    c.realism.contentV2 = true;
    c.realism.latentReactions = true;
    c.realism.sessionDynamics = true;
    c.scheduling.openStaggerSeconds = 3600.0;     // opens spread over the first hour
    c.scheduling.returnDelayMeanSeconds = 3600.0; // ~1h baseline return
    return c;
}

std::string readFile(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

nlohmann::json readJson(const fs::path &p) {
    std::ifstream in(p);
    return nlohmann::json::parse(in);
}

} // namespace

// --- All four §6 groups + the event-mode block, coherent and well-formed ------------------------
TEST(EventRunnerPipelineTest, ProducesAllFourGroupsAndEventModeAdditions) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p18_pipeline";
    fs::remove_all(root);
    ExperimentRunner runner(eventConfig(), root);
    const ExperimentResult r = runner.run();

    // The event runner actually ran the world: requests were served and impressions consumed as an
    // OUTCOME of the timelines (interactions_per_user is ignored in event mode).
    EXPECT_GT(r.requestCount, 0u);
    EXPECT_GT(r.impressionCount, 0u);

    // Four §6 metric groups (D22), each as its own file/block.
    EXPECT_TRUE(fs::exists(r.directory / "recommendation_metrics.csv")); // engagement (V1)
    EXPECT_TRUE(fs::exists(r.directory / "welfare_metrics.csv"));        // hidden welfare
    EXPECT_TRUE(fs::exists(r.directory / "welfare_archetype_metrics.csv"));
    EXPECT_TRUE(fs::exists(r.directory / "session_health.csv"));    // session health
    EXPECT_TRUE(fs::exists(r.directory / "diversity_metrics.csv")); // recommendation quality
    EXPECT_TRUE(fs::exists(r.directory / "retrieval_metrics.csv"));
    ASSERT_TRUE(r.welfare.configured);
    ASSERT_TRUE(r.sessionHealth.configured);

    // Event-mode additions carried in-process (package C reads these) and in summary.json.
    ASSERT_TRUE(r.eventMode.configured);
    EXPECT_GT(r.eventMode.eventCount, 0u);
    EXPECT_DOUBLE_EQ(r.eventMode.simulatedDays, 6.0 * 3600.0 / 86400.0); // 0.25 simulated days
    EXPECT_GT(r.eventMode.sessionsPerSimulatedDay, 0.0);

    const nlohmann::json summary = readJson(r.directory / "summary.json");
    ASSERT_TRUE(summary.contains("event_mode"));
    const nlohmann::json &em = summary.at("event_mode");
    for (const char *key :
         {"scheduler", "event_count", "event_log_digest", "simulated_days",
          "sessions_per_simulated_day", "mean_concurrent_online", "return_delay_seconds"}) {
        EXPECT_TRUE(em.contains(key)) << "event_mode block missing key: " << key;
    }
    EXPECT_EQ(em.at("scheduler").get<std::string>(), "event_queue");
    EXPECT_EQ(em.at("event_log_digest").get<uint64_t>(), r.eventMode.eventLogDigest);
    // All four §6 groups are still indexed in metric_groups; no aggregate score (D22).
    ASSERT_TRUE(summary.contains("metric_groups"));
    EXPECT_EQ(summary.at("metric_groups").at("session_health").at("status").get<std::string>(),
              "live");
    EXPECT_FALSE(summary.contains("aggregate_score"));
}

// --- Determinism: same seed => identical digest AND byte-identical deterministic CSVs ------------
TEST(EventRunnerPipelineTest, SameSeedReproducesDigestAndCsvs) {
    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_p18_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_p18_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);
    const ExperimentResult a = ExperimentRunner(eventConfig(), rootA).run();
    const ExperimentResult b = ExperimentRunner(eventConfig(), rootB).run();

    // The event-log digest (D20 tripwire) and event count are identical.
    EXPECT_EQ(a.eventMode.eventLogDigest, b.eventMode.eventLogDigest);
    EXPECT_EQ(a.eventMode.eventCount, b.eventMode.eventCount);
    EXPECT_GT(a.eventMode.eventCount, 0u);

    // Deterministic CSVs reproduce byte-for-byte (the four §6 groups' files).
    for (const char *f :
         {"welfare_metrics.csv", "welfare_archetype_metrics.csv", "session_health.csv",
          "recommendation_metrics.csv", "diversity_metrics.csv"}) {
        EXPECT_EQ(readFile(a.directory / f), readFile(b.directory / f))
            << "non-deterministic file: " << f;
    }
}

// --- Independent timelines: users open, exit, and return at different simulated times ------------
// Two small worlds that differ ONLY in open-stagger width. Both exercise the independence evidence;
// comparing them decisively shows entry times are staggered, not lockstep. (Small to stay cheap.)
TEST(EventRunnerPipelineTest, UsersRunOnIndependentTimelines) {
    ExperimentConfig spread = eventConfig();
    spread.simulation.users = 100;
    spread.simulation.reels = 800;
    spread.simulation.horizonSeconds = 3.0 * 3600.0;
    spread.scheduling.openStaggerSeconds = 3.0 * 3600.0; // opens over the whole window
    ExperimentConfig synced = spread;
    synced.scheduling.openStaggerSeconds = 1.0; // everyone opens at t~=0

    const fs::path rootS = fs::path(::testing::TempDir()) / "rr_p18_spread";
    const fs::path rootY = fs::path(::testing::TempDir()) / "rr_p18_synced";
    fs::remove_all(rootS);
    fs::remove_all(rootY);
    const ExperimentResult spreadR = ExperimentRunner(spread, rootS).run();
    const ExperimentResult syncedR = ExperimentRunner(synced, rootY).run();

    // Sessions and returns actually happen (users exit mid-run and get scheduled to come back), so
    // the timelines are genuinely multi-session — impossible if everyone sat on one shared
    // timeline.
    EXPECT_GT(spreadR.sessionHealth.sessions, 0u);
    EXPECT_GT(spreadR.eventMode.returnCount, 0u);

    // Concurrent-online occupancy is strictly between 0 and 1: users are NOT all online at once
    // (staggered opens + independent exits/returns), yet some are always active.
    EXPECT_GT(spreadR.eventMode.meanConcurrentOnline, 0.0);
    EXPECT_LT(spreadR.eventMode.meanConcurrentOnline, 1.0);

    // Decisive open-stagger check: opens spread across the WHOLE horizon give lower mean concurrent
    // occupancy than a synchronized world where every user opens at t~=0 — direct evidence that
    // entry times are independent, not lockstep.
    EXPECT_LT(spreadR.eventMode.meanConcurrentOnline, syncedR.eventMode.meanConcurrentOnline)
        << "staggered opens should spread users out more than synchronized opens";
}

// --- Legacy round-robin on the SAME config (minus the scheduler) is untouched --------------------
TEST(EventRunnerPipelineTest, LegacyRoundRobinPathStillWorksAndCarriesNoEventMode) {
    ExperimentConfig c = eventConfig();
    c.simulation.scheduler = "round_robin"; // the only change
    c.simulation.interactionsPerUser = 15;  // round_robin uses this (event mode ignores it)
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p18_legacy";
    fs::remove_all(root);
    const ExperimentResult r = ExperimentRunner(c, root).run();

    // The legacy runner produces its usual gate-on output...
    EXPECT_GT(r.impressionCount, 0u);
    EXPECT_TRUE(r.welfare.configured);
    EXPECT_TRUE(r.sessionHealth.configured);
    // ...and NONE of the event-mode additions leak into the round-robin path (byte-identity, D17).
    EXPECT_FALSE(r.eventMode.configured);
    const nlohmann::json summary = readJson(r.directory / "summary.json");
    EXPECT_FALSE(summary.contains("event_mode"))
        << "round-robin summary.json must not carry an event_mode block (D17 byte-identity)";
}
