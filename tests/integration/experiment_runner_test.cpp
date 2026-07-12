#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/simulation/dataset_generator.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

// Tiny but non-degenerate experiment: enough users/reels to exercise sessions, multiple rounds,
// and oracle sampling, while finishing in milliseconds.
ExperimentConfig tinyConfig(RecommendationAlgorithm algo) {
    ExperimentConfig c;
    c.simulation.seed = 7;
    c.simulation.users = 30;
    c.simulation.reels = 300;
    c.simulation.creators = 10;
    c.simulation.topics = 8;
    c.simulation.dimensions = 32;
    c.simulation.interactionsPerUser = 10;
    c.recommendation.feedSize = 5; // -> ceil(10/5) = 2 rounds
    c.recommendation.vectorCandidates = 100;
    c.evaluation.oracleSampleRate = 0.5;
    c.algorithm = algo;
    return c;
}

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

// Data rows (excluding the header) of a CSV file.
std::vector<std::vector<std::string>> readCsvRows(const fs::path &p) {
    std::ifstream f(p);
    std::string line;
    std::vector<std::vector<std::string>> rows;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) {
            header = false;
            continue;
        }
        if (!line.empty()) {
            rows.push_back(splitCsv(line));
        }
    }
    return rows;
}

// The (unparsed) header line of a CSV file.
std::string readCsvHeader(const fs::path &p) {
    std::ifstream f(p);
    std::string line;
    std::getline(f, line);
    return line;
}

// The independently-computed cold-start estimate<->hidden alignment for a config: mean over all
// users of cos(prior, hiddenPreference) where prior is the global-average hidden preference. This
// is exactly what the frozen arm should report every round (estimates never move off the prior).
double coldStartAlignment(const ExperimentConfig &c) {
    GeneratedDataset ds = generateDataset(c.simulation, c.simulation.seed);
    const Embedding prior = globalAveragePreference(ds.hiddenStates);
    double sum = 0.0;
    for (const HiddenUserState &h : ds.hiddenStates) {
        sum += dot(prior, h.hiddenPreference);
    }
    return ds.hiddenStates.empty() ? 0.0 : sum / static_cast<double>(ds.hiddenStates.size());
}

} // namespace

// (a) End-to-end tiny run produces every §26 file with sane contents.
TEST(ExperimentRunnerTest, ProducesAllOutputFilesWithSaneContents) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_files";
    fs::remove_all(root);

    ExperimentRunner runner(tinyConfig(RecommendationAlgorithm::ExactVector), root);
    ExperimentResult result = runner.run();

    // Counts add up.
    EXPECT_EQ(result.userCount, 30u);
    EXPECT_EQ(result.reelCount, 300u);
    EXPECT_EQ(result.rounds.size(), 2u);
    EXPECT_EQ(result.requestCount, 30u * 2u); // one request per user per round
    EXPECT_EQ(result.impressionCount, 30u * 10u);

    // All §26 files exist.
    for (const char *name :
         {"config.json", "summary.json", "retrieval_metrics.csv", "recommendation_metrics.csv",
          "learning_curve.csv", "regret_curve.csv", "latency_metrics.csv", "metadata.json"}) {
        EXPECT_TRUE(fs::exists(result.directory / name)) << "missing " << name;
    }

    // recommendation_metrics.csv: one row per round, rates in [0,1], reward in [-1,1], affinity
    // finite.
    auto recRows = readCsvRows(result.directory / "recommendation_metrics.csv");
    ASSERT_EQ(recRows.size(), 2u);
    for (const auto &row : recRows) {
        ASSERT_EQ(row.size(), 13u);
        for (int col : {4, 5, 6, 7, 8}) { // skip/completion/like/share/follow rates
            const double v = std::stod(row[col]);
            EXPECT_GE(v, 0.0);
            EXPECT_LE(v, 1.0);
        }
        const double rewardPerImp = std::stod(row[10]);
        EXPECT_GE(rewardPerImp, -1.0);
        EXPECT_LE(rewardPerImp, 1.0);
        EXPECT_TRUE(std::isfinite(std::stod(row[12]))); // mean_true_affinity
    }

    // learning_curve.csv, regret_curve.csv, and retrieval_metrics.csv also one row per round.
    EXPECT_EQ(readCsvRows(result.directory / "learning_curve.csv").size(), 2u);
    EXPECT_EQ(readCsvRows(result.directory / "regret_curve.csv").size(), 2u);
    EXPECT_EQ(readCsvRows(result.directory / "retrieval_metrics.csv").size(), 2u);

    // latency_metrics.csv: header + one row per stage (total, retrieval, ranking, reranking).
    EXPECT_EQ(readCsvRows(result.directory / "latency_metrics.csv").size(), 4u);

    // Overall metric ranges.
    EXPECT_GE(result.overall.completionRate, 0.0);
    EXPECT_LE(result.overall.completionRate, 1.0);
    EXPECT_TRUE(std::isfinite(result.overall.meanTrueAffinity));
    EXPECT_GT(result.sampledRequestCount, 0u); // 0.5 rate over 60 requests: overwhelmingly likely
}

// (b) Full-run determinism: same config + seed twice ⇒ byte-identical deterministic outputs.
// latency_metrics.csv, metadata.json, and summary.json's timing subsection are excluded (wall
// clock, D9/TDD 24.6).
TEST(ExperimentRunnerTest, FullRunDeterminismByteIdentical) {
    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_exp_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_exp_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);

    // Random exercises the recommender rng stream, the strongest determinism case.
    ExperimentRunner runnerA(tinyConfig(RecommendationAlgorithm::Random), rootA);
    ExperimentRunner runnerB(tinyConfig(RecommendationAlgorithm::Random), rootB);
    ExperimentResult a = runnerA.run();
    ExperimentResult b = runnerB.run();

    for (const char *name : {"config.json", "retrieval_metrics.csv", "recommendation_metrics.csv",
                             "learning_curve.csv", "regret_curve.csv"}) {
        EXPECT_EQ(readFile(a.directory / name), readFile(b.directory / name))
            << name << " differs between two same-seed runs";
    }
}

// (c) Live retrieval self-check: an exact recommender's ANN index IS exact ground truth, so live
// recall must be exactly 1.0 and distance error exactly 0.0 (TDD 18.1 wiring self-check). Sampling
// every request (rate 1.0) guarantees samples at smoke scale, and the retrieval index search
// registers a measurable per-stage latency.
TEST(ExperimentRunnerTest, ExactVectorLiveRetrievalIsPerfect) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_retrieval_exact";
    fs::remove_all(root);

    ExperimentConfig config = tinyConfig(RecommendationAlgorithm::ExactVector);
    config.evaluation.retrievalSampleRate = 1.0; // every request sampled -> samples guaranteed
    ExperimentRunner runner(config, root);
    const ExperimentResult result = runner.run();

    EXPECT_TRUE(result.retrievalApplicable);
    EXPECT_GT(result.retrievalSampleCount, 0u);
    EXPECT_DOUBLE_EQ(result.retrievalRecallAt10, 1.0);
    EXPECT_DOUBLE_EQ(result.retrievalRecallAt50, 1.0);
    EXPECT_DOUBLE_EQ(result.retrievalDistanceError, 0.0);
    // The retrieval index search is on the timed path -> a positive p50 stage latency.
    EXPECT_GT(result.retrievalLatency.p50Ms, 0.0);
    EXPECT_DOUBLE_EQ(result.rerankingLatency.p50Ms, 0.0); // no reranking stage

    // summary.json exposes the same in its retrieval section.
    std::ifstream in(result.directory / "summary.json");
    nlohmann::json j;
    in >> j;
    const auto &r = j.at("retrieval");
    EXPECT_TRUE(r.at("applicable").get<bool>());
    EXPECT_GT(r.at("sampled_requests").get<size_t>(), 0u);
    EXPECT_DOUBLE_EQ(r.at("recall_at_10").get<double>(), 1.0);
    EXPECT_DOUBLE_EQ(r.at("recall_at_50").get<double>(), 1.0);
    EXPECT_DOUBLE_EQ(r.at("mean_distance_error").get<double>(), 0.0);
}

// (d) Non-vector algorithm: no retrieval samples, a note is written, and retrieval_metrics.csv is
// still emitted (uniform layout) with zero-sample rows.
TEST(ExperimentRunnerTest, NonVectorAlgorithmWritesZeroSampleRetrieval) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_retrieval_random";
    fs::remove_all(root);

    ExperimentConfig config = tinyConfig(RecommendationAlgorithm::Random);
    config.evaluation.retrievalSampleRate = 1.0; // even at rate 1.0, no vector index -> 0 samples
    ExperimentRunner runner(config, root);
    const ExperimentResult result = runner.run();

    EXPECT_FALSE(result.retrievalApplicable);
    EXPECT_EQ(result.retrievalSampleCount, 0u);

    // File exists with one zero-sample row per round.
    ASSERT_TRUE(fs::exists(result.directory / "retrieval_metrics.csv"));
    auto rows = readCsvRows(result.directory / "retrieval_metrics.csv");
    ASSERT_EQ(rows.size(), 2u);
    for (const auto &row : rows) {
        ASSERT_EQ(row.size(), 5u);
        EXPECT_EQ(std::stoul(row[1]), 0u);        // samples
        EXPECT_DOUBLE_EQ(std::stod(row[2]), 0.0); // recall_at_10
        EXPECT_DOUBLE_EQ(std::stod(row[3]), 0.0); // recall_at_50
        EXPECT_DOUBLE_EQ(std::stod(row[4]), 0.0); // mean_distance_error
    }

    // summary note flags the non-vector case.
    std::ifstream in(result.directory / "summary.json");
    nlohmann::json j;
    in >> j;
    EXPECT_FALSE(j.at("retrieval").at("applicable").get<bool>());
    EXPECT_FALSE(j.at("retrieval").at("note").get<std::string>().empty());
}

// (e) retrieval_metrics.csv is deterministic with NON-trivial values: exact index searches are
// deterministic, so an exact_vector run at a fixed sample rate is byte-identical across seeds.
TEST(ExperimentRunnerTest, RetrievalMetricsDeterministicForVectorAlgorithm) {
    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_exp_retr_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_exp_retr_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);

    ExperimentConfig config = tinyConfig(RecommendationAlgorithm::ExactVector);
    config.evaluation.retrievalSampleRate = 0.5; // partial sampling -> exercises the rng alignment
    ExperimentRunner runnerA(config, rootA);
    ExperimentRunner runnerB(config, rootB);
    const ExperimentResult a = runnerA.run();
    const ExperimentResult b = runnerB.run();

    EXPECT_GT(a.retrievalSampleCount, 0u);
    EXPECT_EQ(a.retrievalSampleCount, b.retrievalSampleCount); // aligned "retrieval" rng stream
    EXPECT_EQ(readFile(a.directory / "retrieval_metrics.csv"),
              readFile(b.directory / "retrieval_metrics.csv"));
}

// (f) learning_curve.csv schema (Phase 7): exact header, one data row per round, and the
// interactions_per_user column equals min((round+1)*feedSize, interactionsPerUser).
TEST(ExperimentRunnerTest, LearningCurveSchema) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_learncurve_schema";
    fs::remove_all(root);

    const ExperimentConfig config = tinyConfig(RecommendationAlgorithm::ExactVector);
    ExperimentRunner runner(config, root);
    const ExperimentResult result = runner.run();

    const fs::path csv = result.directory / "learning_curve.csv";
    EXPECT_EQ(
        readCsvHeader(csv),
        "round,interactions_per_user,mean_reward_per_impression,mean_estimated_hidden_cosine");

    const auto rows = readCsvRows(csv);
    ASSERT_EQ(rows.size(), result.rounds.size()); // one data row per round
    const size_t feedSize = config.recommendation.feedSize;
    const size_t budget = config.simulation.interactionsPerUser;
    for (size_t r = 0; r < rows.size(); ++r) {
        ASSERT_EQ(rows[r].size(), 4u);
        EXPECT_EQ(std::stoul(rows[r][0]), r); // round index
        EXPECT_EQ(std::stoul(rows[r][1]), std::min((r + 1) * feedSize, budget));
        EXPECT_TRUE(std::isfinite(std::stod(rows[r][2]))); // reward per impression
        const double cosine = std::stod(rows[r][3]);       // estimate<->hidden alignment
        EXPECT_GE(cosine, -1.0001);
        EXPECT_LE(cosine, 1.0001);
    }
}

// (g) Frozen arm (learning.enabled == false): the three per-user preference vectors never move off
// the cold-start prior. ExperimentRunner encapsulates the users, so the externally observable
// projection of "estimatedPreference is frozen" is the learning curve's alignment column: it must
// be CONSTANT across rounds AND equal to the independently-computed cold-start alignment.
TEST(ExperimentRunnerTest, FrozenArmAlignmentIsConstantAtColdStart) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_frozen";
    fs::remove_all(root);

    ExperimentConfig config = tinyConfig(RecommendationAlgorithm::ExactVector);
    config.learning.enabled = false;
    ExperimentRunner runner(config, root);
    const ExperimentResult result = runner.run();

    EXPECT_FALSE(result.learningEnabled);

    const double expected = coldStartAlignment(config);
    const auto rows = readCsvRows(result.directory / "learning_curve.csv");
    ASSERT_GE(rows.size(), 2u); // multiple rounds so "constant across rounds" is meaningful

    const std::string firstCosine = rows.front()[3];
    for (const auto &row : rows) {
        ASSERT_EQ(row.size(), 4u);
        EXPECT_EQ(row[3], firstCosine) << "frozen alignment column varies across rounds";
        EXPECT_NEAR(std::stod(row[3]), expected, 1e-5); // ... and equals the cold-start prior value
    }
    EXPECT_NEAR(result.finalEstimatedHiddenCosine, expected, 1e-5);
}

// (h) Learning arm (learning.enabled == true): under online updates most users' estimatedPreference
// moves off the cold-start prior, so the learning curve's alignment column becomes NON-constant.
//
// EXPECTED FAIL in THIS worktree: the OnlineUserStateUpdater here is Package A's no-op stub, so no
// preference vector ever changes and the column stays constant -> the ASSERT below fails. This is
// the intended signal that the updater is inert; the test is correct and passes once the real
// updater lands at integration. It fails ONLY because the vectors never change (no crash, no other
// assertion): the cold-start estimates are still unit-length, so ranking/feature extraction and the
// rest of the run behave exactly as in the frozen arm.
TEST(ExperimentRunnerTest, LearningArmAlignmentChangesOverTime) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_learning";
    fs::remove_all(root);

    ExperimentConfig config = tinyConfig(RecommendationAlgorithm::ExactVector);
    config.learning.enabled = true;
    config.simulation.interactionsPerUser = 20;
    config.recommendation.feedSize =
        5; // -> 4 rounds, enough to see convergence with a real updater
    ExperimentRunner runner(config, root);
    const ExperimentResult result = runner.run();

    EXPECT_TRUE(result.learningEnabled);

    const auto rows = readCsvRows(result.directory / "learning_curve.csv");
    ASSERT_GE(rows.size(), 2u);
    // The alignment column must vary: at least one round differs from the first. Against the no-op
    // stub it is constant and this fails (expected until integration with Package A's updater).
    bool anyDifferent = false;
    for (const auto &row : rows) {
        ASSERT_EQ(row.size(), 4u);
        if (row[3] != rows.front()[3]) {
            anyDifferent = true;
        }
    }
    EXPECT_TRUE(anyDifferent)
        << "estimate<->hidden alignment is constant across rounds: the OnlineUserStateUpdater did "
           "not change any user's estimatedPreference. EXPECTED against Package A's no-op stub; "
           "passes once the real updater is integrated.";
}

// (i) learning_curve.csv determinism: same config + seed twice ⇒ byte-identical. Deterministic
// even against the no-op stub (num() fixed precision; the updater consumes no rng). Mirrors the
// existing byte-identical determinism tests.
TEST(ExperimentRunnerTest, LearningCurveDeterministicByteIdentical) {
    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_exp_learn_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_exp_learn_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);

    ExperimentConfig config = tinyConfig(RecommendationAlgorithm::HnswRanker);
    config.learning.enabled = true;
    ExperimentRunner runnerA(config, rootA);
    ExperimentRunner runnerB(config, rootB);
    const ExperimentResult a = runnerA.run();
    const ExperimentResult b = runnerB.run();

    EXPECT_EQ(readFile(a.directory / "learning_curve.csv"),
              readFile(b.directory / "learning_curve.csv"));
}

// (j) No-injection regression contract (Phase 8): with newUsers == newReels == 0 (the default) the
// two Phase-8 cold-start files are NOT written and summary.json carries no `cold_start` key, so the
// output layout is byte-identical to a pre-Phase-8 run. (The existing determinism suite pins the
// CSV *content* half of the contract; this pins the "no new files / no new keys" half.)
TEST(ExperimentRunnerTest, NoInjectionOmitsColdStartOutputs) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_no_injection";
    fs::remove_all(root);

    const ExperimentConfig config = tinyConfig(RecommendationAlgorithm::ExactVector);
    ASSERT_EQ(config.simulation.newUsers, 0u);
    ASSERT_EQ(config.simulation.newReels, 0u);
    ExperimentRunner runner(config, root);
    const ExperimentResult result = runner.run();

    EXPECT_FALSE(result.coldStart.configured);
    EXPECT_FALSE(fs::exists(result.directory / "new_user_curve.csv"));
    EXPECT_FALSE(fs::exists(result.directory / "new_reel_exposure.csv"));

    std::ifstream in(result.directory / "summary.json");
    nlohmann::json j;
    in >> j;
    EXPECT_FALSE(j.contains("cold_start")) << "summary.json must have no cold_start key when off";
}

// (k) enableExploration is config-driven since Phase 8 but read by NO existing recommender, so
// toggling exploration.enabled leaves a non-exploration algorithm's deterministic metric CSVs
// byte-identical TODAY. This documents the flip's safety; it still passes post-merge for every
// non-exploration algorithm. (config.json differs by the exploration.enabled value, so it is
// intentionally excluded from the comparison.)
TEST(ExperimentRunnerTest, ExplorationFlagFlipIsInertForNonExplorationAlgorithm) {
    const fs::path rootOn = fs::path(::testing::TempDir()) / "rr_exp_explore_on";
    const fs::path rootOff = fs::path(::testing::TempDir()) / "rr_exp_explore_off";
    fs::remove_all(rootOn);
    fs::remove_all(rootOff);

    ExperimentConfig on = tinyConfig(RecommendationAlgorithm::HnswRanker);
    on.exploration.enabled = true;
    ExperimentConfig off = tinyConfig(RecommendationAlgorithm::HnswRanker);
    off.exploration.enabled = false;

    ExperimentRunner runnerOn(on, rootOn);
    ExperimentRunner runnerOff(off, rootOff);
    const ExperimentResult a = runnerOn.run();
    const ExperimentResult b = runnerOff.run();

    for (const char *name : {"retrieval_metrics.csv", "recommendation_metrics.csv",
                             "learning_curve.csv", "regret_curve.csv"}) {
        EXPECT_EQ(readFile(a.directory / name), readFile(b.directory / name))
            << name << " changed when exploration.enabled flipped (should be inert)";
    }
}

// (l) Diversity metrics (Phase 9, TDD 18.4): diversity_metrics.csv is written for EVERY run with
// one row per round, and every value sits in its valid range (topics/creators in [1, feedSize], HHI
// in (0, 1], intra-list similarity in [-1, 1], repetition rate 0). The summary.json diversity block
// mirrors the overall figures.
TEST(ExperimentRunnerTest, DiversityMetricsEmittedWithValidRanges) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_diversity";
    fs::remove_all(root);

    const ExperimentConfig config = tinyConfig(RecommendationAlgorithm::HnswRanker);
    ExperimentRunner runner(config, root);
    const ExperimentResult result = runner.run();

    ASSERT_TRUE(fs::exists(result.directory / "diversity_metrics.csv"));
    EXPECT_EQ(readCsvHeader(result.directory / "diversity_metrics.csv"),
              "round,mean_unique_topics,mean_unique_creators,mean_intra_list_similarity,"
              "mean_topic_hhi,mean_creator_hhi,repetition_rate");

    auto rows = readCsvRows(result.directory / "diversity_metrics.csv");
    ASSERT_EQ(rows.size(), result.rounds.size()); // one row per round
    const double feedSize = static_cast<double>(config.recommendation.feedSize);
    for (const auto &row : rows) {
        ASSERT_EQ(row.size(), 7u);
        const double topics = std::stod(row[1]);
        const double creators = std::stod(row[2]);
        const double sim = std::stod(row[3]);
        const double topicHhi = std::stod(row[4]);
        const double creatorHhi = std::stod(row[5]);
        const double repetition = std::stod(row[6]);
        EXPECT_GE(topics, 1.0);
        EXPECT_LE(topics, feedSize);
        EXPECT_GE(creators, 1.0);
        EXPECT_LE(creators, feedSize);
        EXPECT_GE(sim, -1.0);
        EXPECT_LE(sim, 1.0);
        EXPECT_GT(topicHhi, 0.0);
        EXPECT_LE(topicHhi, 1.0);
        EXPECT_GT(creatorHhi, 0.0);
        EXPECT_LE(creatorHhi, 1.0);
        EXPECT_DOUBLE_EQ(repetition, 0.0);
    }

    // In-memory overall figures line up with the summary.json diversity block.
    EXPECT_EQ(result.diversityFeedCount, result.requestCount);
    EXPECT_GE(result.meanUniqueTopics, 1.0);
    EXPECT_LE(result.meanUniqueTopics, feedSize);
    EXPECT_EQ(result.totalRepetitions, 0u);
    EXPECT_DOUBLE_EQ(result.repetitionRate, 0.0);

    std::ifstream in(result.directory / "summary.json");
    nlohmann::json j;
    in >> j;
    ASSERT_TRUE(j.contains("diversity"));
    const auto &d = j.at("diversity");
    EXPECT_EQ(d.at("feeds").get<size_t>(), result.requestCount);
    EXPECT_DOUBLE_EQ(d.at("mean_unique_topics").get<double>(), result.meanUniqueTopics);
    EXPECT_DOUBLE_EQ(d.at("mean_intra_list_similarity").get<double>(),
                     result.meanIntraListSimilarity);
    EXPECT_EQ(d.at("repetition_total").get<size_t>(), 0u);
    EXPECT_DOUBLE_EQ(d.at("repetition_rate").get<double>(), 0.0);
}

// (m) diversity_metrics.csv is deterministic: same config + seed twice -> byte-identical file
// (extends the determinism contract to the Phase 9 file).
TEST(ExperimentRunnerTest, DiversityMetricsDeterministicByteIdentical) {
    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_exp_div_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_exp_div_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);

    ExperimentRunner runnerA(tinyConfig(RecommendationAlgorithm::HnswRanker), rootA);
    ExperimentRunner runnerB(tinyConfig(RecommendationAlgorithm::HnswRanker), rootB);
    const ExperimentResult a = runnerA.run();
    const ExperimentResult b = runnerB.run();

    EXPECT_EQ(readFile(a.directory / "diversity_metrics.csv"),
              readFile(b.directory / "diversity_metrics.csv"))
        << "diversity_metrics.csv differs between two same-seed runs";
}

// (n) Live verification of the Phase 9 "duplicate/repetitive content eliminated" exit criterion:
// the repetition rate is EXACTLY 0 across the whole run (no seen reel is re-served and no feed
// contains a duplicate id), for every round and overall.
TEST(ExperimentRunnerTest, RepetitionRateIsZeroAcrossRun) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_exp_div_norepeat";
    fs::remove_all(root);

    ExperimentRunner runner(tinyConfig(RecommendationAlgorithm::HnswRanker), root);
    const ExperimentResult result = runner.run();

    EXPECT_EQ(result.totalRepetitions, 0u);
    EXPECT_DOUBLE_EQ(result.repetitionRate, 0.0);
    for (const RoundMetrics &r : result.rounds) {
        EXPECT_EQ(r.repetitionCount, 0u) << "round " << r.round << " served a repeat";
        EXPECT_DOUBLE_EQ(r.repetitionRate, 0.0);
    }
    for (const auto &row : readCsvRows(result.directory / "diversity_metrics.csv")) {
        ASSERT_EQ(row.size(), 7u);
        EXPECT_DOUBLE_EQ(std::stod(row[6]), 0.0); // repetition_rate column
    }
}

// (o) enableDiversity is config-driven since Phase 9 but read by NO existing recommender, so
// toggling diversity.enabled leaves an existing algorithm's deterministic metric CSVs (including
// diversity_metrics.csv itself, since the feeds are identical) byte-identical TODAY. This proves
// the flag alone changes nothing for existing algorithms. (config.json differs by the
// diversity.enabled value, so it is intentionally excluded from the comparison.)
TEST(ExperimentRunnerTest, DiversityFlagFlipIsInertForExistingAlgorithm) {
    const fs::path rootOn = fs::path(::testing::TempDir()) / "rr_exp_div_on";
    const fs::path rootOff = fs::path(::testing::TempDir()) / "rr_exp_div_off";
    fs::remove_all(rootOn);
    fs::remove_all(rootOff);

    ExperimentConfig on = tinyConfig(RecommendationAlgorithm::HnswRanker);
    on.diversity.enabled = true;
    ExperimentConfig off = tinyConfig(RecommendationAlgorithm::HnswRanker);
    off.diversity.enabled = false;

    ExperimentRunner runnerOn(on, rootOn);
    ExperimentRunner runnerOff(off, rootOff);
    const ExperimentResult a = runnerOn.run();
    const ExperimentResult b = runnerOff.run();

    for (const char *name : {"retrieval_metrics.csv", "recommendation_metrics.csv",
                             "diversity_metrics.csv", "learning_curve.csv", "regret_curve.csv"}) {
        EXPECT_EQ(readFile(a.directory / name), readFile(b.directory / name))
            << name << " changed when diversity.enabled flipped (should be inert)";
    }
}
