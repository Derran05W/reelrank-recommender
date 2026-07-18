#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/user.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/drift_scheduler.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/simulator.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

// -------------------------------------------------------------------------------------------------
// EXPECTED-FAIL PROTOCOL (Phase 10, Package B builds against the FROZEN DriftScheduler interface
// while a sibling package implements src/simulation/drift_scheduler.cpp). The current stub makes
// maybeApply / everApplies / inCohort inert (no-ops returning false). Tests whose PASS depends on
// real drift *behaviour* are written to the frozen contract and marked below with
// "[EXPECTED-FAIL until sibling drift_scheduler.cpp lands]"; the integrator verifies them at merge.
// Tests that only exercise wiring / output structure / determinism PASS now against the stub.
// -------------------------------------------------------------------------------------------------

std::string readFile(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> splitCsv(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

// Header + data rows of a CSV, each split into comma fields.
std::vector<std::vector<std::string>> readCsvRows(const fs::path &p) {
    std::ifstream f(p);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) {
            rows.push_back(splitCsv(line));
        }
    }
    return rows;
}

// normalize(sum_i weight_i * topic_centre_i) -- the frozen application math (drift_scheduler.hpp).
Embedding weightedTarget(const std::vector<DriftTopicWeight> &mix, const std::vector<Topic> &topics,
                         std::size_t dim) {
    Embedding target(dim, 0.0f);
    for (const DriftTopicWeight &w : mix) {
        const Embedding &centre = topics[w.topic].centre;
        for (std::size_t d = 0; d < dim; ++d) {
            target[d] += static_cast<float>(w.weight) * centre[d];
        }
    }
    normalize(target);
    return target;
}

double cosine(const Embedding &a, const Embedding &b) { return dot(a, b); }

// Small hnsw_ranker config with online learning ON; `interactionsPerUser` / `feedSize` chosen so
// there are several rounds and a drift can be scheduled a couple rounds in.
ExperimentConfig runnerConfig() {
    ExperimentConfig c;
    c.simulation.seed = 13;
    c.simulation.users = 40;
    c.simulation.reels = 400;
    c.simulation.creators = 12;
    c.simulation.topics = 12;
    c.simulation.dimensions = 32;
    c.simulation.interactionsPerUser = 20; // feed 5 -> 4 rounds
    c.recommendation.feedSize = 5;
    c.recommendation.vectorCandidates = 120;
    c.evaluation.oracleSampleRate = 0.5;
    c.algorithm = RecommendationAlgorithm::HnswRanker;
    return c;
}

// Whole-population drift at interaction 10 (feed 5 -> driftRound 2): rounds 0,1 are strictly
// pre-drift, rounds 2,3 post-drift. Mix on two high-index topics disjoint from most users.
DriftConfig runnerDrift() {
    DriftConfig d;
    DriftEvent e;
    e.atInteraction = 10;
    e.cohortLo = 0.0;
    e.cohortHi = 1.0;
    e.topicMix = {DriftTopicWeight{10, 1.0}, DriftTopicWeight{11, 1.0}};
    d.events = {e};
    return d;
}

} // namespace

// (a) Scheduler + Simulator micro-loop (no full runner): drive impressions by hand, calling
// maybeApply before each step exactly as the harness does. Asserts the hidden preference changes at
// EXACTLY the configured interaction, is unit-length, equals the frozen weighted-target math, and
// that the recommender-visible estimatedPreference is untouched at the drift instant.
// [EXPECTED-FAIL until sibling drift_scheduler.cpp lands: the stub's maybeApply never mutates
// hidden
//  state, so the "changed at drift" assertions fail.]
TEST(DriftAdaptationTest, MicroLoopHiddenDriftFiresAtExactInteraction) {
    SimulationConfig sc;
    sc.seed = 5;
    sc.users = 8;
    sc.reels = 120;
    sc.creators = 6;
    sc.topics = 12;
    sc.dimensions = 32;
    sc.interactionsPerUser = 12;

    GeneratedDataset ds = generateDataset(sc, sc.seed);
    // Cold-start prior so estimatedPreference is a real (non-empty) recommender-visible vector.
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

    User &user = ds.users[0];
    HiddenUserState &hidden = ds.hiddenStates[0];
    const Embedding original = hidden.hiddenPreference;

    // Choose a drift mix of topics NOT among this user's preferred topics, guaranteeing a genuine
    // change under the real scheduler.
    std::vector<uint32_t> foreign;
    for (uint32_t t = 0; t < sc.topics && foreign.size() < 2; ++t) {
        const bool preferred =
            std::find(hidden.preferredTopics.begin(), hidden.preferredTopics.end(), TopicId{t}) !=
            hidden.preferredTopics.end();
        if (!preferred) {
            foreign.push_back(t);
        }
    }
    ASSERT_EQ(foreign.size(), 2u);

    DriftConfig dc;
    DriftEvent e;
    e.atInteraction = 4;
    e.cohortLo = 0.0;
    e.cohortHi = 1.0;
    e.topicMix = {DriftTopicWeight{foreign[0], 1.0}, DriftTopicWeight{foreign[1], 1.0}};
    dc.events = {e};

    const DriftScheduler drift(dc, ds.topics);
    const Embedding expected = weightedTarget(e.topicMix, ds.topics, sc.dimensions);
    // Guard: the drift target is genuinely different from the original preference.
    ASSERT_LT(cosine(original, expected), 0.999) << "test setup: drift mix is too aligned already";

    Simulator sim(BehaviourConfig{}, RewardConfig{}, forkRng(sc.seed, "behaviour"),
                  LearningConfig{}.recentWindow, RankingConfig{}.trendingHalfLifeSeconds);

    for (int i = 0; i < 8; ++i) {
        const uint32_t before = static_cast<uint32_t>(user.totalInteractions);
        const Embedding hiddenBefore = hidden.hiddenPreference;
        const Embedding estBefore = user.estimatedPreference;

        drift.maybeApply(hidden, before);

        // The drift touches ONLY hidden state: the recommender-visible estimate is never perturbed.
        EXPECT_EQ(user.estimatedPreference, estBefore)
            << "maybeApply must not change the recommender-visible estimate (impression " << i
            << ")";

        if (before < e.atInteraction) {
            EXPECT_EQ(hidden.hiddenPreference, hiddenBefore)
                << "hidden preference must not drift before the configured interaction";
            EXPECT_EQ(hidden.hiddenPreference, original);
        } else {
            // At and after the drift interaction the hidden preference is the frozen weighted
            // target.
            ASSERT_EQ(hidden.hiddenPreference.size(), expected.size());
            for (std::size_t d = 0; d < expected.size(); ++d) {
                EXPECT_NEAR(hidden.hiddenPreference[d], expected[d], 1e-5)
                    << "drifted hidden preference component " << d;
            }
            EXPECT_TRUE(isValid(hidden.hiddenPreference))
                << "drifted preference must be unit-length";
            EXPECT_EQ(hidden.preferredTopics,
                      (std::vector<TopicId>{TopicId{foreign[0]}, TopicId{foreign[1]}}));
        }

        Reel &reel = ds.reels[static_cast<std::size_t>(i)];
        const Creator &creator = ds.creators[reel.creatorId.value];
        sim.step(user, hidden, reel, creator);
    }
}

// (b) Full runner, drift ON vs OFF at the same seed: rounds strictly BEFORE driftRound are
// byte-identical in the first four (legacy) learning_curve columns, and at least one later row
// differs once the drift has moved the hidden state.
// [EXPECTED-FAIL until sibling drift_scheduler.cpp lands: with the inert stub the drift-on and
//  drift-off simulations are identical, so no later row differs.]
TEST(DriftAdaptationTest, FullRunnerDriftDivergesAfterDriftRound) {
    const fs::path rootOn = fs::path(::testing::TempDir()) / "rr_drift_on";
    const fs::path rootOff = fs::path(::testing::TempDir()) / "rr_drift_off";
    fs::remove_all(rootOn);
    fs::remove_all(rootOff);

    ExperimentConfig on = runnerConfig();
    on.drift = runnerDrift();
    ExperimentConfig off = runnerConfig(); // no drift events

    const ExperimentResult rOn = ExperimentRunner(on, rootOn).run();
    const ExperimentResult rOff = ExperimentRunner(off, rootOff).run();

    const long driftRound = rOn.adaptation.driftRound;
    ASSERT_GE(driftRound, 1) << "config must leave pre-drift rounds";

    const auto rowsOn = readCsvRows(rOn.directory / "learning_curve.csv");
    const auto rowsOff = readCsvRows(rOff.directory / "learning_curve.csv");
    ASSERT_EQ(rowsOn.size(), rowsOff.size());
    ASSERT_GT(rowsOn.size(), 1u);

    // Pre-drift rounds: the first four columns match exactly (drift has not fired yet).
    for (std::size_t i = 1; i < rowsOn.size(); ++i) { // i=0 is the header
        const long round = std::stol(rowsOn[i][0]);
        if (round < driftRound) {
            for (int col = 0; col < 4; ++col) {
                EXPECT_EQ(rowsOn[i][col], rowsOff[i][col])
                    << "pre-drift round " << round << " column " << col << " must be identical";
            }
        }
    }

    // Post-drift: at least one row's first-four columns differ (the drift changed the outcome).
    bool anyDiff = false;
    for (std::size_t i = 1; i < rowsOn.size(); ++i) {
        const long round = std::stol(rowsOn[i][0]);
        if (round >= driftRound) {
            for (int col = 0; col < 4; ++col) {
                if (rowsOn[i][col] != rowsOff[i][col]) {
                    anyDiff = true;
                }
            }
        }
    }
    EXPECT_TRUE(anyDiff) << "drift must change some post-drift round's reward/alignment";
}

// (c) Drift OFF: the learning_curve keeps the legacy 4-column header and summary.json has no
// `adaptation` key. MUST PASS NOW against the stub (configured() is real and returns false).
TEST(DriftAdaptationTest, DriftOffIsByteIdenticalLegacyStructure) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_drift_off_legacy";
    fs::remove_all(root);

    const ExperimentResult r = ExperimentRunner(runnerConfig(), root).run();
    EXPECT_FALSE(r.adaptation.configured);

    const auto rows = readCsvRows(r.directory / "learning_curve.csv");
    ASSERT_FALSE(rows.empty());
    const std::vector<std::string> expectedHeader = {"round", "interactions_per_user",
                                                     "mean_reward_per_impression",
                                                     "mean_estimated_hidden_cosine"};
    EXPECT_EQ(rows.front(), expectedHeader);
    for (const auto &row : rows) {
        EXPECT_EQ(row.size(), 4u) << "drift-off learning_curve must stay 4 columns";
    }

    const nlohmann::json summary = nlohmann::json::parse(readFile(r.directory / "summary.json"));
    EXPECT_FALSE(summary.contains("adaptation"))
        << "no adaptation block when drift is unconfigured";
}

// (d) Determinism: a drift-ON run at the same seed twice yields byte-identical deterministic CSVs.
// The 8-column learning_curve and the adaptation block are exercised now (configured() is real);
// the byte-identical guarantee holds regardless of whether drift application is stubbed or real.
TEST(DriftAdaptationTest, DriftOnIsDeterministic) {
    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_drift_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_drift_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);

    ExperimentConfig cfg = runnerConfig();
    cfg.drift = runnerDrift();

    const ExperimentResult a = ExperimentRunner(cfg, rootA).run();
    const ExperimentResult b = ExperimentRunner(cfg, rootB).run();

    ASSERT_TRUE(a.adaptation.configured);
    // The drift-on learning_curve carries the four extra cohort columns.
    const auto rows = readCsvRows(a.directory / "learning_curve.csv");
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows.front().size(), 8u) << "drift-on learning_curve must have 8 columns";

    for (const char *f : {"learning_curve.csv", "regret_curve.csv", "recommendation_metrics.csv"}) {
        EXPECT_EQ(readFile(a.directory / f), readFile(b.directory / f))
            << f << " must be byte-identical across same-seed runs";
    }
}
