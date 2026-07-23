// Phase 22 training-log DETERMINISM / ROTATION / GATE-OFF suite (package A; contracts §7 row A).
//   * same seed twice => byte-identical training_log/ tree (files + contents);
//   * rotation kicks in under a tiny log_max_rows_per_file (contiguous part files, row-capped);
//   * gate-off (training_log=false) => NO training_log dir AND the event-digest + deterministic
//   CSVs
//     are byte-identical to a gate-ON run (so logging + the survey draw ZERO simulation rng — the
//     "explicit-feedback" stream and the pinned rng-free sampling never perturb the V1 world);
//   * logging ON but survey OFF => no survey.csv and still digest-identical to fully off.

#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace rr;

namespace {

namespace fs = std::filesystem;

// Base tiny event world (no logging). Individual tests toggle the learning_v2 gates.
ExperimentConfig baseConfig(uint64_t seed = 20260722) {
    ExperimentConfig c;
    c.simulation.seed = seed;
    c.simulation.users = 100;
    c.simulation.reels = 800;
    c.simulation.creators = 30;
    c.simulation.topics = 8;
    c.simulation.dimensions = 16;
    c.simulation.scheduler = "event_queue";
    c.simulation.horizonSeconds = 3.0 * 3600.0;
    c.recommendation.feedSize = 10;
    c.recommendation.vectorCandidates = 100;
    c.evaluation.oracleSampleRate = 0.05;
    c.evaluation.retrievalSampleRate = 0.02;
    c.algorithm = RecommendationAlgorithm::HnswRanker;
    c.realism.contentV2 = true;
    c.realism.latentReactions = true;
    c.realism.sessionDynamics = true;
    c.scheduling.openStaggerSeconds = 1800.0;
    c.scheduling.returnDelayMeanSeconds = 1800.0;
    return c;
}

std::string readFile(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::size_t dataRows(const fs::path &csv) {
    std::ifstream in(csv);
    std::string line;
    std::size_t n = 0;
    while (std::getline(in, line)) {
        ++n;
    }
    return n == 0 ? 0 : n - 1; // minus header
}

// Sorted list of file NAMES under `dir` whose name starts with `prefix`.
std::vector<std::string> partNames(const fs::path &dir, const std::string &prefix) {
    std::vector<std::string> names;
    for (const auto &entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

} // namespace

// --- Same seed twice => byte-identical training_log/ tree
// ------------------------------------------
TEST(TrainingLogDeterminismTest, SameSeedByteIdenticalLogTree) {
    ExperimentConfig c = baseConfig();
    c.learningV2.trainingLog = true;
    c.learningV2.logSampleRate = 0.5;
    c.learningV2.logPoolSampleRate = 0.3;
    c.learningV2.logMaxRowsPerFile = 300; // small => multiple parts, so rotation naming is compared
    c.learningV2.survey.enabled = true;
    c.learningV2.survey.sampleRate = 0.4;

    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_p22_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_p22_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);
    const fs::path logA = ExperimentRunner(c, rootA).run().directory / "training_log";
    const fs::path logB = ExperimentRunner(c, rootB).run().directory / "training_log";

    // Same single files, byte-for-byte.
    for (const char *f : {"schema.json", "requests.csv", "survey.csv"}) {
        EXPECT_EQ(readFile(logA / f), readFile(logB / f)) << "non-deterministic file: " << f;
    }
    // Same rotation part sets, byte-for-byte.
    for (const char *prefix : {"candidates-part", "outcomes-part"}) {
        const auto namesA = partNames(logA, prefix);
        const auto namesB = partNames(logB, prefix);
        ASSERT_EQ(namesA, namesB) << "part file set differs for " << prefix;
        ASSERT_FALSE(namesA.empty());
        for (const auto &name : namesA) {
            EXPECT_EQ(readFile(logA / name), readFile(logB / name))
                << "non-deterministic: " << name;
        }
    }
}

// --- Rotation: tiny row limit => contiguous, row-capped part files
// ---------------------------------
TEST(TrainingLogDeterminismTest, RotationProducesContiguousCappedParts) {
    ExperimentConfig c = baseConfig();
    c.learningV2.trainingLog = true;
    c.learningV2.logSampleRate = 1.0;
    c.learningV2.logPoolSampleRate = 1.0;
    c.learningV2.logMaxRowsPerFile = 200;

    const fs::path root = fs::path(::testing::TempDir()) / "rr_p22_rotation";
    fs::remove_all(root);
    const fs::path log = ExperimentRunner(c, root).run().directory / "training_log";

    const auto candParts = partNames(log, "candidates-part");
    ASSERT_GE(candParts.size(), 2u) << "rotation did not kick in";
    // Contiguous 0000, 0001, ... and each non-final part is exactly full.
    for (std::size_t i = 0; i < candParts.size(); ++i) {
        std::ostringstream want;
        want << "candidates-part" << std::setw(4) << std::setfill('0') << i << ".csv";
        EXPECT_EQ(candParts[i], want.str());
        const std::size_t rows = dataRows(log / candParts[i]);
        EXPECT_LE(rows, 200u);
        if (i + 1 < candParts.size()) {
            EXPECT_EQ(rows, 200u) << "non-final part " << candParts[i] << " should be full";
        }
    }
}

// --- Gate-off byte-identity + stream discipline
// ---------------------------------------------------
TEST(TrainingLogDeterminismTest, GateOffByteIdenticalToGateOn) {
    ExperimentConfig off = baseConfig();
    ExperimentConfig on = baseConfig();
    on.learningV2.trainingLog = true;
    on.learningV2.logSampleRate = 1.0;
    on.learningV2.logPoolSampleRate = 1.0;
    on.learningV2.survey.enabled = true;
    on.learningV2.survey.sampleRate = 1.0;

    const fs::path rootOff = fs::path(::testing::TempDir()) / "rr_p22_gate_off";
    const fs::path rootOn = fs::path(::testing::TempDir()) / "rr_p22_gate_on";
    fs::remove_all(rootOff);
    fs::remove_all(rootOn);
    const ExperimentResult rOff = ExperimentRunner(off, rootOff).run();
    const ExperimentResult rOn = ExperimentRunner(on, rootOn).run();

    // Gate-off emits NO training_log dir; gate-on does.
    EXPECT_FALSE(fs::exists(rOff.directory / "training_log"));
    EXPECT_TRUE(fs::exists(rOn.directory / "training_log"));

    // The simulation is byte-identical: the event-log digest matches, so logging + the survey drew
    // ZERO simulation rng (the pinned sampling draws none; the survey draws only on its own
    // stream).
    EXPECT_EQ(rOn.eventMode.eventLogDigest, rOff.eventMode.eventLogDigest);
    EXPECT_EQ(rOn.eventMode.eventCount, rOff.eventMode.eventCount);
    EXPECT_GT(rOn.eventMode.eventCount, 0u);

    // Every deterministic §6 metric CSV reproduces byte-for-byte across the gate.
    for (const char *f :
         {"welfare_metrics.csv", "welfare_archetype_metrics.csv", "session_health.csv",
          "recommendation_metrics.csv", "diversity_metrics.csv"}) {
        EXPECT_EQ(readFile(rOff.directory / f), readFile(rOn.directory / f))
            << "gate flipped a deterministic file: " << f;
    }
}

// --- Logging ON but survey OFF: no survey.csv, and still digest-identical to fully off
// -------------
TEST(TrainingLogDeterminismTest, LoggingOnSurveyOffDrawsNoExplicitFeedback) {
    ExperimentConfig off = baseConfig();
    ExperimentConfig logOnly = baseConfig();
    logOnly.learningV2.trainingLog = true;
    logOnly.learningV2.logSampleRate = 1.0;
    logOnly.learningV2.logPoolSampleRate = 1.0;
    // survey stays disabled (default)

    const fs::path rootOff = fs::path(::testing::TempDir()) / "rr_p22_log_off_ref";
    const fs::path rootLog = fs::path(::testing::TempDir()) / "rr_p22_log_only";
    fs::remove_all(rootOff);
    fs::remove_all(rootLog);
    const ExperimentResult rOff = ExperimentRunner(off, rootOff).run();
    const ExperimentResult rLog = ExperimentRunner(logOnly, rootLog).run();

    const fs::path log = rLog.directory / "training_log";
    EXPECT_TRUE(fs::exists(log));
    EXPECT_FALSE(fs::exists(log / "survey.csv")) << "survey.csv written despite survey disabled";
    EXPECT_TRUE(fs::exists(log / "requests.csv"));

    // Logging on but zero explicit-feedback draws => the world is byte-identical to fully off.
    EXPECT_EQ(rLog.eventMode.eventLogDigest, rOff.eventMode.eventLogDigest);
}
