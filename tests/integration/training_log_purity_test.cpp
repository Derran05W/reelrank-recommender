// Phase 22 training-log PURITY audit (package A; contracts §4, V2 §7 mandated). Runs a TINY gate-on
// event sim with training_log + survey enabled, then audits the EMITTED FILES (not the structs):
//   (a) every CSV header column is a member of the frozen allowlist in training_log_schema.hpp;
//   (b) a forbidden-substring scan on the feature/outcome table headers
//   (latent/satisfaction/regret/
//       archetype/trust/hidden/fatigue/plasticity/tolerance) — survey.csv exempted for its columns;
//   (c) schema.json echoes the frozen version + column allowlists.
// It also demonstrates that position/exposure/eligibility metadata is persisted end-to-end (Tier 5
// acceptance 4): the emitted candidates.csv carries both shown impressions (with feed positions)
// and pool-only candidates (position -1), plus exploration provenance.

#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "rr/learning_v2/training_log_schema.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

// A tiny gate-on log-world: HnswRanker (a ranked recommender, so the ranked capture path is
// exercised), full V2 stack, and BOTH log rates + the survey at 1.0 so every request's full pool
// and every shown impression are logged (maximal coverage for the audit).
ExperimentConfig logConfig(uint64_t seed = 20260722) {
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
    c.learningV2.trainingLog = true;
    c.learningV2.logSampleRate = 1.0;
    c.learningV2.logPoolSampleRate = 1.0;
    c.learningV2.survey.enabled = true;
    c.learningV2.survey.sampleRate = 1.0;
    c.learningV2.survey.noiseSd = 0.35;
    return c;
}

std::vector<std::string> readLines(const fs::path &p) {
    std::ifstream in(p);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::vector<std::string> splitCsv(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        out.push_back(cell);
    }
    return out;
}

template <std::size_t N>
std::set<std::string> allowSet(const std::array<std::string_view, N> &cols) {
    std::set<std::string> s;
    for (const auto c : cols) {
        s.emplace(c);
    }
    return s;
}

std::set<std::string> headerSet(const fs::path &csv) {
    const auto lines = readLines(csv);
    std::set<std::string> s;
    if (!lines.empty()) {
        for (const auto &c : splitCsv(lines[0])) {
            s.insert(c);
        }
    }
    return s;
}

// The full candidates header allowlist: the frozen prefix UNION the frozen feature columns.
std::set<std::string> candidatesAllowSet() {
    std::set<std::string> s = allowSet(learning_v2::kCandidatesPrefixColumns);
    for (const auto c : learning_v2::kFeatureColumns) {
        s.emplace(c);
    }
    return s;
}

const std::vector<std::string> kForbidden = {"latent",    "satisfaction", "regret",
                                             "archetype", "trust",        "hidden",
                                             "fatigue",   "plasticity",   "tolerance"};

void expectNoForbiddenInHeader(const fs::path &csv) {
    const auto lines = readLines(csv);
    ASSERT_FALSE(lines.empty()) << csv;
    for (const auto &col : splitCsv(lines[0])) {
        for (const auto &bad : kForbidden) {
            EXPECT_EQ(col.find(bad), std::string::npos)
                << "forbidden substring '" << bad << "' in header column '" << col << "' of "
                << csv;
        }
    }
}

} // namespace

TEST(TrainingLogPurityTest, EmittedFilesAreHiddenFreeAndSchemaConformant) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p22_purity";
    fs::remove_all(root);
    const ExperimentResult r = ExperimentRunner(logConfig(), root).run();
    const fs::path log = r.directory / "training_log";

    ASSERT_GT(r.requestCount, 0u);
    ASSERT_GT(r.impressionCount, 0u);

    // The four log tables + schema exist.
    ASSERT_TRUE(fs::exists(log / "schema.json"));
    ASSERT_TRUE(fs::exists(log / "requests.csv"));
    ASSERT_TRUE(fs::exists(log / "candidates-part0000.csv"));
    ASSERT_TRUE(fs::exists(log / "outcomes-part0000.csv"));
    ASSERT_TRUE(fs::exists(log / "survey.csv"));

    // (a) Every header column is a member of the exact frozen allowlist (over ALL part files).
    EXPECT_EQ(headerSet(log / "requests.csv"), allowSet(learning_v2::kRequestsColumns));
    EXPECT_EQ(headerSet(log / "survey.csv"), allowSet(learning_v2::kSurveyColumns));
    const std::set<std::string> candAllow = candidatesAllowSet();
    const std::set<std::string> outAllow = allowSet(learning_v2::kOutcomesColumns);
    for (const auto &entry : fs::directory_iterator(log)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("candidates-part", 0) == 0) {
            EXPECT_EQ(headerSet(entry.path()), candAllow) << name;
            expectNoForbiddenInHeader(entry.path()); // (b)
        } else if (name.rfind("outcomes-part", 0) == 0) {
            EXPECT_EQ(headerSet(entry.path()), outAllow) << name;
            expectNoForbiddenInHeader(entry.path()); // (b)
        }
    }

    // (c) schema.json echoes the frozen schema.
    std::ifstream schemaIn(log / "schema.json");
    const nlohmann::json schema = nlohmann::json::parse(schemaIn);
    EXPECT_EQ(schema.at("schema_version").get<int>(), learning_v2::kSchemaVersion);
    std::vector<std::string> featureEcho =
        schema.at("feature_columns").get<std::vector<std::string>>();
    ASSERT_EQ(featureEcho.size(), learning_v2::kFeatureColumns.size());
    for (std::size_t i = 0; i < featureEcho.size(); ++i) {
        EXPECT_EQ(featureEcho[i], learning_v2::kFeatureColumns[i]);
    }
    // The one hidden-derived table is clearly labeled.
    const auto hidden = schema.at("hidden_derived_tables").get<std::vector<std::string>>();
    ASSERT_EQ(hidden.size(), 1u);
    EXPECT_EQ(hidden[0], "survey");
    // Table column echoes match the emitted headers.
    std::set<std::string> schemaCand(schema.at("tables").at("candidates").begin(),
                                     schema.at("tables").at("candidates").end());
    EXPECT_EQ(schemaCand, candAllow);
}

// --- Position / exposure / eligibility metadata persisted end-to-end (Tier 5 acceptance 4)
// --------
TEST(TrainingLogPurityTest, PoolShownPositionsAndProvenancePersisted) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p22_persist";
    fs::remove_all(root);
    const ExperimentResult r = ExperimentRunner(logConfig(43), root).run();
    const fs::path log = r.directory / "training_log";

    // Walk every candidates part: confirm BOTH shown impressions (with a feed position >= 0) and
    // pool-only candidates (position -1, shown 0) were logged, and exploration_flag is a 0/1
    // column.
    bool sawShown = false;
    bool sawPoolOnly = false;
    bool sawExplorationColumn = false;
    for (const auto &entry : fs::directory_iterator(log)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("candidates-part", 0) != 0) {
            continue;
        }
        const auto lines = readLines(entry.path());
        for (std::size_t i = 1; i < lines.size(); ++i) {
            const auto c = splitCsv(lines[i]);
            ASSERT_GE(c.size(), 9u);
            const std::string shown = c[3];
            const int position = std::stoi(c[4]);
            const std::string explorationFlag = c[6];
            if (shown == "1") {
                EXPECT_GE(position, 0);
                sawShown = true;
            } else if (shown == "0") {
                EXPECT_EQ(position, -1);
                sawPoolOnly = true;
            }
            if (explorationFlag == "0" || explorationFlag == "1") {
                sawExplorationColumn = true;
            }
        }
    }
    EXPECT_TRUE(sawShown) << "no shown impressions logged with feed positions";
    EXPECT_TRUE(sawPoolOnly) << "no pool-only candidates logged (full-pool exposure missing)";
    EXPECT_TRUE(sawExplorationColumn);

    // A requests.csv row proves the pool is strictly larger than the shown feed (full-pool
    // capture).
    const auto reqLines = readLines(log / "requests.csv");
    ASSERT_GT(reqLines.size(), 1u);
    bool poolBiggerThanShown = false;
    for (std::size_t i = 1; i < reqLines.size(); ++i) {
        const auto row = splitCsv(reqLines[i]);
        ASSERT_EQ(row.size(), learning_v2::kRequestsColumns.size());
        const long poolSize = std::stol(row[6]);
        const long shownCount = std::stol(row[7]);
        if (poolSize > shownCount) {
            poolBiggerThanShown = true;
        }
    }
    EXPECT_TRUE(poolBiggerThanShown);
}
