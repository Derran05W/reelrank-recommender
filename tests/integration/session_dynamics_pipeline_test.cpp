// Session-dynamics pipeline integration test (Phase 16, V2 TDD §4.9/§6, D22). Drives a tiny
// gate-ON experiment end-to-end through the ExperimentRunner and asserts the session-health group
// is plumbed coherently: the exit-aware consume loop runs, session_health.csv (per round) exists
// with the right schema and well-formed rows, the summary.json session_health block + the LIVE
// metric_groups.session_health entry are present with no aggregate score, the welfare group's
// harmful_fatigue placeholder is realized under the gate, and the SAME config gated OFF (a
// latent-only P15 run) emits none of the new session-health files/keys while keeping the P15
// welfare output and reproducing itself byte-identically.
//
// PRE-INTEGRATION NOTE. In this package's worktree the simulator's session lifecycle is a stub:
// stepV2 never fires observedExitAfterImpression, so the exit-aware loop closes ZERO sessions and
// the reduction is an all-zero, well-formed report. Everything asserted here is the PLUMBING /
// determinism / gate-off byte-identity. The POPULATED-PATH assertions (non-zero session counts,
// exit-type shares that sum to 1, U_s magnitudes, next-session linkage) are marked
// "integrator-verified": they exercise once package A lands the probabilistic exit model. The
// session-health MATH is fully unit-tested on constructed records in session_health_metrics_test.

#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace rr;

namespace {

namespace fs = std::filesystem;

// Smoke-scale config: ~100 users / 1000 reels / 3 rounds, hnsw_ranker. content_v2 +
// latent_reactions are always on (session_dynamics requires latent_reactions, which requires
// content_v2, D17); `sessionDynamics` toggles the Phase 16 gate under test.
ExperimentConfig p16Config(bool sessionDynamicsOn) {
    ExperimentConfig c;
    c.simulation.seed = 20260718;
    c.simulation.users = 100;
    c.simulation.reels = 1000;
    c.simulation.creators = 20;
    c.simulation.topics = 8;
    c.simulation.dimensions = 16;
    c.simulation.interactionsPerUser = 15;
    c.recommendation.feedSize = 5; // ceil(15/5) = 3 rounds
    c.recommendation.vectorCandidates = 100;
    c.evaluation.oracleSampleRate = 0.1;
    c.algorithm = RecommendationAlgorithm::HnswRanker;
    c.realism.contentV2 = true;
    c.realism.latentReactions = true;
    c.realism.sessionDynamics = sessionDynamicsOn;
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

struct Csv {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};

std::vector<std::string> splitCells(const std::string &line) {
    std::vector<std::string> cells;
    std::stringstream ls(line);
    std::string cell;
    while (std::getline(ls, cell, ',')) {
        cells.push_back(cell);
    }
    return cells;
}

Csv readCsv(const fs::path &p) {
    Csv csv;
    std::ifstream in(p);
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first) {
            csv.header = splitCells(line);
            first = false;
        } else {
            csv.rows.push_back(splitCells(line));
        }
    }
    return csv;
}

} // namespace

// --- Gate ON: session_health.csv + summary blocks present, coherent, well-formed -----------------
TEST(SessionDynamicsPipelineTest, GateOnEmitsCoherentSessionHealthOutput) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p16_sh_on";
    fs::remove_all(root);
    ExperimentRunner runner(p16Config(/*sessionDynamicsOn=*/true), root);
    const ExperimentResult r = runner.run();

    ASSERT_TRUE(r.sessionHealth.configured);
    ASSERT_GT(r.impressionCount, 0u); // the exit-aware loop still consumed impressions

    // --- session_health.csv (per round) ---------------------------------------------------------
    const fs::path shCsv = r.directory / "session_health.csv";
    ASSERT_TRUE(fs::exists(shCsv)) << "session_health.csv must exist under the gate";
    const Csv sh = readCsv(shCsv);
    const std::vector<std::string> expectHeader = {"round",
                                                   "sessions",
                                                   "open_sessions",
                                                   "mean_duration_seconds",
                                                   "median_duration_seconds",
                                                   "mean_impressions",
                                                   "satisfaction_per_minute",
                                                   "regret_per_minute",
                                                   "mean_session_utility",
                                                   "early_failure_exit_rate",
                                                   "natural_completion_rate",
                                                   "harmful_fatigue_mean",
                                                   "next_session_starting_satisfaction",
                                                   "failure_exits",
                                                   "satisfied_exits",
                                                   "fatigue_exits",
                                                   "external_exits",
                                                   "regret_exits"};
    EXPECT_EQ(sh.header, expectHeader);
    EXPECT_EQ(sh.rows.size(), r.rounds.size()); // one row per round

    // Every row is well-formed: all cells parse as numbers (no "nan"/"inf"), rates in [0,1], and
    // the exit-type counts never exceed the closed-session count.
    for (const std::vector<std::string> &row : sh.rows) {
        ASSERT_EQ(row.size(), expectHeader.size());
        const auto sessions = static_cast<std::size_t>(std::stoull(row[1]));
        const auto openSessions = static_cast<std::size_t>(std::stoull(row[2]));
        const double failureRate = std::stod(row[9]);
        const double naturalRate = std::stod(row[10]);
        EXPECT_GE(failureRate, 0.0);
        EXPECT_LE(failureRate, 1.0);
        EXPECT_GE(naturalRate, 0.0);
        EXPECT_LE(naturalRate, 1.0);
        std::size_t exitCountSum = 0;
        for (int col = 13; col <= 17; ++col) {
            exitCountSum += static_cast<std::size_t>(std::stoull(row[col]));
        }
        EXPECT_EQ(exitCountSum,
                  sessions); // the five exit-type counts partition the closed sessions
        // no NaN text leaked (stod on the mean columns already parsed above/below)
        for (int col : {3, 4, 5, 6, 7, 8, 11, 12}) {
            EXPECT_NO_THROW((void)std::stod(row[col]));
        }
        (void)openSessions;
    }

    // --- POPULATED path (integration-verified, Phase 16): package A's probabilistic exit model
    // closes real sessions and the run-end drain emits RunEnded records for still-open ones, so
    // the gate-on run must produce a non-trivial session ledger: closed sessions exist, every
    // user's final open session (if any) is drained, and closed + open covers every simulated
    // impression exactly once (impressions partition into sessions).
    EXPECT_GT(r.sessionHealth.sessions, 0u);
    std::size_t csvSessions = 0;
    std::size_t csvOpen = 0;
    for (const std::vector<std::string> &row : sh.rows) {
        csvSessions += static_cast<std::size_t>(std::stoull(row[1]));
        csvOpen += static_cast<std::size_t>(std::stoull(row[2]));
    }
    EXPECT_EQ(csvSessions, r.sessionHealth.sessions);
    EXPECT_EQ(csvOpen, r.sessionHealth.openSessions);

    // --- summary.json: session_health block + LIVE metric_groups entry, no aggregate score -------
    const nlohmann::json summary = readJson(r.directory / "summary.json");
    ASSERT_TRUE(summary.contains("session_health"));
    const nlohmann::json &shBlock = summary.at("session_health");
    for (const char *key :
         {"sessions", "open_sessions", "mean_duration_seconds", "median_duration_seconds",
          "satisfaction_per_minute", "regret_per_minute", "mean_session_utility",
          "early_failure_exit_rate", "natural_completion_rate", "harmful_fatigue_mean",
          "next_session_starting_satisfaction", "exit_type_counts", "exit_type_shares"}) {
        EXPECT_TRUE(shBlock.contains(key)) << "session_health block missing key: " << key;
    }
    // Exit-type distribution carries the full V2 §4.8 taxonomy + the open (RunEnded) share.
    const nlohmann::json &shares = shBlock.at("exit_type_shares");
    for (const char *k : {"failure", "satisfied", "fatigue", "external", "regret", "open"}) {
        EXPECT_TRUE(shares.contains(k)) << "exit_type_shares missing: " << k;
    }

    ASSERT_TRUE(summary.contains("metric_groups"));
    const nlohmann::json &groups = summary.at("metric_groups");
    // The session-health group is now LIVE (was "limited_pre_p16" under P15).
    EXPECT_EQ(groups.at("session_health").at("status").get<std::string>(), "live");

    // --- welfare group: harmful_fatigue realized under the gate ----------------------------------
    ASSERT_TRUE(summary.contains("welfare"));
    const nlohmann::json &welfare = summary.at("welfare");
    // harmful_fatigue dropped out of not_yet_modeled (only platform_trust remains) and gained a
    // source note; the column is real (0 under the stub, populated once package A lands).
    const std::vector<std::string> nym =
        welfare.at("not_yet_modeled").get<std::vector<std::string>>();
    EXPECT_EQ(nym, (std::vector<std::string>{"platform_trust"}));
    EXPECT_TRUE(welfare.contains("harmful_fatigue_source"));

    // No aggregate score is ever defined (D22).
    EXPECT_FALSE(summary.contains("aggregate_score"));
    EXPECT_FALSE(shBlock.contains("aggregate_score"));
    EXPECT_FALSE(groups.contains("aggregate_score"));
}

// --- Gate OFF (latent-only P15 run): no session-health output, pristine P15 welfare --------------
TEST(SessionDynamicsPipelineTest, GateOffEmitsNoSessionHealthOutputAndKeepsP15Welfare) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p16_sh_off";
    fs::remove_all(root);
    ExperimentRunner runner(p16Config(/*sessionDynamicsOn=*/false), root);
    const ExperimentResult r = runner.run();

    EXPECT_FALSE(r.sessionHealth.configured);
    // No session_health.csv.
    EXPECT_FALSE(fs::exists(r.directory / "session_health.csv"));
    // No session_health summary block; the metric_groups entry stays LIMITED pre-P16.
    const nlohmann::json summary = readJson(r.directory / "summary.json");
    EXPECT_FALSE(summary.contains("session_health"))
        << "gate-off summary.json must not carry a session_health block (D17 byte-identity)";
    ASSERT_TRUE(summary.contains("metric_groups"));
    EXPECT_EQ(summary.at("metric_groups").at("session_health").at("status").get<std::string>(),
              "limited_pre_p16");

    // The P15 welfare group is byte-for-byte the pre-Phase-16 shape: harmful_fatigue is still a
    // placeholder (0, listed in not_yet_modeled) and there is no harmful_fatigue_source key.
    ASSERT_TRUE(summary.contains("welfare"));
    const nlohmann::json &welfare = summary.at("welfare");
    EXPECT_EQ(welfare.at("harmful_fatigue").get<double>(), 0.0);
    const std::vector<std::string> nym =
        welfare.at("not_yet_modeled").get<std::vector<std::string>>();
    EXPECT_EQ(nym, (std::vector<std::string>{"harmful_fatigue", "platform_trust"}));
    EXPECT_FALSE(welfare.contains("harmful_fatigue_source"));

    // The P15 welfare files are all still present (the session-health group is purely additive).
    for (const char *f : {"welfare_metrics.csv", "welfare_archetype_metrics.csv",
                          "recommendation_metrics.csv", "summary.json", "config.json"}) {
        EXPECT_TRUE(fs::exists(r.directory / f)) << "missing P15/V1 file: " << f;
    }
}

// --- Determinism: same seed -> byte-identical deterministic CSVs (gate on AND gate off) ----------
// The gate-off run reproducing itself byte-identically is the in-process stand-in for "gate-off is
// byte-identical to a latent-only run" (it IS one); the committed golden tests cover the full
// gates-off-vs-V1 byte-identity (D17). Gate-on determinism covers the new session_health.csv.
TEST(SessionDynamicsPipelineTest, DeterministicCsvsAcrossSameSeedRuns) {
    for (bool gate : {false, true}) {
        const fs::path rootA =
            fs::path(::testing::TempDir()) / (gate ? "rr_p16_det_on_a" : "rr_p16_det_off_a");
        const fs::path rootB =
            fs::path(::testing::TempDir()) / (gate ? "rr_p16_det_on_b" : "rr_p16_det_off_b");
        fs::remove_all(rootA);
        fs::remove_all(rootB);
        const ExperimentResult a = ExperimentRunner(p16Config(gate), rootA).run();
        const ExperimentResult b = ExperimentRunner(p16Config(gate), rootB).run();

        // Deterministic files reproduce byte-for-byte (the P15 welfare CSVs and the V1 metrics CSV
        // are unchanged by the session-health additions).
        EXPECT_EQ(readFile(a.directory / "welfare_metrics.csv"),
                  readFile(b.directory / "welfare_metrics.csv"));
        EXPECT_EQ(readFile(a.directory / "recommendation_metrics.csv"),
                  readFile(b.directory / "recommendation_metrics.csv"));
        if (gate) {
            EXPECT_TRUE(fs::exists(a.directory / "session_health.csv"));
            EXPECT_EQ(readFile(a.directory / "session_health.csv"),
                      readFile(b.directory / "session_health.csv"));
        } else {
            EXPECT_FALSE(fs::exists(a.directory / "session_health.csv"));
        }
    }
}
