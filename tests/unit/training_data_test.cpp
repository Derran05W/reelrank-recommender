// Phase 22 package B — training_data unit tests: CSV join (shown-only), rotation part-file and
// header-name (reorder / extra column) tolerance, deterministic temporal + user-disjoint splits,
// target extraction, the majority-source key, and the hand-checkable offline-eval metrics +
// eval-row schema (contracts §2/§5).

#include "rr/learning_v2/training_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "learning_v2_test_fixture.hpp"
#include "rr/core/pinned_hash.hpp"
#include "rr/learning_v2/training_log_schema.hpp"

using namespace rr::learning_v2;
namespace fs = std::filesystem;

namespace {

int countFields(const std::string &line) {
    return static_cast<int>(std::count(line.begin(), line.end(), ',')) + 1;
}

} // namespace

// ---- reader
// --------------------------------------------------------------------------------------

TEST(TrainingDataReaderTest, JoinsShownRowsOnlyWithMetadata) {
    const fs::path dir = test::makeTempDir("join");
    test::PlantedLogParams p;
    p.requests = 40;
    p.shownPerRequest = 3;
    p.poolOnlyPerRequest = 2;
    test::writePlantedLog(dir, p);

    const Dataset ds = loadTrainingLog(dir, /*withSurvey=*/false);
    EXPECT_EQ(ds.shownCandidates, 40u * 3u);   // pool-only rows never counted
    EXPECT_EQ(ds.joinedWithOutcome, 40u * 3u); // every shown row has an outcome
    EXPECT_EQ(ds.droppedNoRequest, 0u);
    EXPECT_EQ(ds.rows.size(), 40u * 3u);
    for (const Example &e : ds.rows) {
        EXPECT_NE(e.requestId, 0u);
        EXPECT_NE(e.userId, 0u);
        EXPECT_NE(e.timestamp, 0u);
    }
    // Rows are canonically ordered by (request_id, reel_id) for reproducible training.
    for (std::size_t i = 1; i < ds.rows.size(); ++i) {
        const auto a = std::make_pair(ds.rows[i - 1].requestId, ds.rows[i - 1].reelId);
        const auto b = std::make_pair(ds.rows[i].requestId, ds.rows[i].reelId);
        EXPECT_LT(a, b);
    }
    fs::remove_all(dir);
}

TEST(TrainingDataReaderTest, TolerantOfRotationPartFiles) {
    const fs::path dir = test::makeTempDir("parts");
    test::PlantedLogParams p;
    p.requests = 20;
    p.shownPerRequest = 2;
    p.poolOnlyPerRequest = 0;
    test::writePlantedLog(dir, p);
    const std::size_t baseRows = loadTrainingLog(dir, false).rows.size();

    // Rename the base tables to rotation parts (base file now absent) — the glob must still find
    // them via the "<base>-partNNNN.csv" pattern.
    fs::rename(dir / "candidates.csv", dir / "candidates-part0000.csv");
    fs::rename(dir / "outcomes.csv", dir / "outcomes-part0000.csv");
    const Dataset ds = loadTrainingLog(dir, false);
    EXPECT_EQ(ds.rows.size(), baseRows);
    EXPECT_GT(baseRows, 0u);
    fs::remove_all(dir);
}

TEST(TrainingDataReaderTest, ParsesByHeaderNameReorderedAndExtraColumns) {
    const fs::path dir = test::makeTempDir("reorder");

    // requests.csv with a scrambled header + an extra column.
    {
        std::ofstream r(dir / "requests.csv");
        r << "junk,timestamp,request_id,user_id\n";
        r << "x,5000,1,10\n";
    }
    // outcomes.csv (minimal required columns; extra column tolerated).
    {
        std::ofstream o(dir / "outcomes.csv");
        o << "request_id,reel_id,watch_ratio,completed,liked,shared,followed,not_interested,"
             "observed_exit_after_impression,extra\n";
        o << "1,100,0.7,1,0,0,0,0,0,zzz\n";
    }
    // candidates.csv: features FIRST (schema order), then the prefix columns, then a junk column.
    {
        std::ofstream c(dir / "candidates.csv");
        for (std::size_t i = 0; i < kFeatureColumns.size(); ++i) {
            c << kFeatureColumns[i] << ',';
        }
        for (std::size_t i = 0; i < kCandidatesPrefixColumns.size(); ++i) {
            c << kCandidatesPrefixColumns[i] << ',';
        }
        c << "junk\n";
        // feature values: similarity=0.11, quality=0.22, rest 0.
        for (std::size_t i = 0; i < kFeatureColumns.size(); ++i) {
            if (kFeatureColumns[i] == "similarity") {
                c << "0.11";
            } else if (kFeatureColumns[i] == "quality") {
                c << "0.22";
            } else {
                c << "0";
            }
            c << ',';
        }
        // prefix values in schema order: request_id,reel_id,pool_rank,shown,position,served_score,
        // exploration_flag,retrieval_sources,retrieval_similarity.
        c << "1,100,0,1,0,1.5,0,vector_hnsw,0.11,";
        c << "JUNK\n";
    }

    const Dataset ds = loadTrainingLog(dir, false);
    ASSERT_EQ(ds.rows.size(), 1u);
    const Example &e = ds.rows[0];
    EXPECT_EQ(e.requestId, 1u);
    EXPECT_EQ(e.reelId, 100u);
    EXPECT_EQ(e.userId, 10u);
    EXPECT_EQ(e.timestamp, 5000u);
    EXPECT_NEAR(e.features[0], 0.11, 1e-9); // similarity mapped by NAME despite reordering
    EXPECT_NEAR(e.features[2], 0.22, 1e-9); // quality
    EXPECT_NEAR(e.servedScore, 1.5, 1e-9);
    EXPECT_EQ(e.retrievalSources, "vector_hnsw");
    EXPECT_EQ(e.completed, 1.0);
    EXPECT_NEAR(e.watchRatio, 0.7, 1e-9);
    fs::remove_all(dir);
}

TEST(TrainingDataReaderTest, ThrowsOnMissingTables) {
    const fs::path dir = test::makeTempDir("empty");
    EXPECT_THROW(loadTrainingLog(dir, false), std::runtime_error);
    fs::remove_all(dir);
}

// ---- splits
// --------------------------------------------------------------------------------------

TEST(TrainingDataSplitTest, TemporalDeterministicAndOrdered) {
    const fs::path dir = test::makeTempDir("temporal");
    test::PlantedLogParams p;
    p.requests = 40;
    p.shownPerRequest = 3;
    p.poolOnlyPerRequest = 0;
    test::writePlantedLog(dir, p);
    const Dataset ds = loadTrainingLog(dir, false);

    const Split a = assignSplit(ds, SplitMode::Temporal);
    const Split b = assignSplit(ds, SplitMode::Temporal);
    EXPECT_EQ(a.train, b.train); // deterministic + seed-independent
    EXPECT_EQ(a.test, b.test);

    EXPECT_EQ(a.train.size(), 32u * 3u); // floor(0.8*40)=32 train requests
    EXPECT_EQ(a.test.size(), 8u * 3u);

    uint64_t maxTrainTs = 0;
    uint64_t minTestTs = UINT64_MAX;
    for (std::size_t i : a.train) {
        maxTrainTs = std::max(maxTrainTs, ds.rows[i].timestamp);
    }
    for (std::size_t i : a.test) {
        minTestTs = std::min(minTestTs, ds.rows[i].timestamp);
    }
    EXPECT_LT(maxTrainTs, minTestTs); // temporal cut: all train precedes all test
    fs::remove_all(dir);
}

TEST(TrainingDataSplitTest, UserDisjointDeterministicHashedAndDisjoint) {
    const fs::path dir = test::makeTempDir("userdisjoint");
    test::PlantedLogParams p;
    p.requests = 200;
    p.shownPerRequest = 2;
    p.poolOnlyPerRequest = 0;
    p.numUsers = 60;
    test::writePlantedLog(dir, p);
    const Dataset ds = loadTrainingLog(dir, false);

    const Split a = assignSplit(ds, SplitMode::UserDisjoint);
    const Split b = assignSplit(ds, SplitMode::UserDisjoint);
    EXPECT_EQ(a.train, b.train);
    EXPECT_EQ(a.test, b.test);

    std::set<uint64_t> trainUsers;
    std::set<uint64_t> testUsers;
    for (std::size_t i : a.train) {
        EXPECT_LT(rr::pinnedHash01(ds.rows[i].userId ^ kSplitSalt), kTrainFraction);
        trainUsers.insert(ds.rows[i].userId);
    }
    for (std::size_t i : a.test) {
        EXPECT_GE(rr::pinnedHash01(ds.rows[i].userId ^ kSplitSalt), kTrainFraction);
        testUsers.insert(ds.rows[i].userId);
    }
    EXPECT_FALSE(a.train.empty());
    EXPECT_FALSE(a.test.empty());
    for (uint64_t u : trainUsers) {
        EXPECT_EQ(testUsers.count(u), 0u); // no user straddles the split
    }
    fs::remove_all(dir);
}

TEST(TrainingDataSplitTest, ParseSplitModeRejectsUnknown) {
    EXPECT_EQ(parseSplitMode("temporal"), SplitMode::Temporal);
    EXPECT_EQ(parseSplitMode("user_disjoint"), SplitMode::UserDisjoint);
    EXPECT_THROW(parseSplitMode("random"), std::invalid_argument);
}

// ---- targets / source key
// ------------------------------------------------------------------------

TEST(TrainingDataTargetTest, SatisfactionRequiresSurveyOthersAlwaysPresent) {
    Example e;
    e.completed = 1.0;
    e.watchRatio = 0.5;
    const TargetSpec *completed = findTarget("completed");
    const TargetSpec *satisfaction = findTarget("satisfaction");
    ASSERT_NE(completed, nullptr);
    ASSERT_NE(satisfaction, nullptr);
    EXPECT_EQ(extractTarget(e, *completed).value(), 1.0);
    EXPECT_FALSE(extractTarget(e, *satisfaction).has_value());
    e.hasSurvey = true;
    e.likert = 4.0;
    EXPECT_EQ(extractTarget(e, *satisfaction).value(), 4.0);
    EXPECT_EQ(allTargets().size(), 8u);
}

TEST(TrainingDataSourceKeyTest, MajorityTokenTieBrokenLexicographically) {
    EXPECT_EQ(majoritySourceKey("vector_hnsw"), "vector_hnsw");
    EXPECT_EQ(majoritySourceKey("a|b|a"), "a");
    EXPECT_EQ(majoritySourceKey("trending exploration exploration"), "exploration");
    EXPECT_EQ(majoritySourceKey("b|a"), "a"); // tie -> lexicographically smallest
    EXPECT_EQ(majoritySourceKey(""), "unknown");
}

// ---- metrics
// -------------------------------------------------------------------------------------

TEST(TrainingDataMetricsTest, RankAucTiesAveraged) {
    EXPECT_NEAR(rankAuc({0.1, 0.2, 0.8, 0.9}, {0, 0, 1, 1}), 1.0, 1e-12);
    EXPECT_NEAR(rankAuc({0.9, 0.8, 0.2, 0.1}, {0, 0, 1, 1}), 0.0, 1e-12);
    EXPECT_NEAR(rankAuc({0.5, 0.5, 0.5, 0.5}, {0, 1, 0, 1}), 0.5, 1e-12); // all tied
    EXPECT_NEAR(rankAuc({1, 2, 3, 4}, {0, 1, 0, 1}), 0.75, 1e-12);
    EXPECT_TRUE(std::isnan(rankAuc({0.1, 0.2}, {1, 1}))); // single class
}

TEST(TrainingDataMetricsTest, LogLossClampedAndRmse) {
    EXPECT_NEAR(logLoss({0.5, 0.5}, {1, 0}), -std::log(0.5), 1e-9);
    EXPECT_NEAR(logLoss({0.0}, {1}), -std::log(1e-6), 1e-9); // clamped
    EXPECT_NEAR(rmse({1, 2, 3}, {1, 2, 3}), 0.0, 1e-12);
    EXPECT_NEAR(rmse({0, 0}, {1, 1}), 1.0, 1e-12);
}

TEST(TrainingDataMetricsTest, CalibrationFitAndBins) {
    const CalibrationFit c = calibrationFit({1, 2, 3}, {2, 4, 6});
    EXPECT_NEAR(c.slope, 2.0, 1e-9);
    EXPECT_NEAR(c.intercept, 0.0, 1e-9);
    EXPECT_TRUE(std::isnan(calibrationFit({5, 5, 5}, {1, 2, 3}).slope)); // constant predictor

    const auto bins = equalCountBins({0.1, 0.2, 0.3, 0.4}, {0, 0, 1, 1}, 2);
    ASSERT_EQ(bins.size(), 2u);
    EXPECT_EQ(bins[0].count, 2);
    EXPECT_EQ(bins[1].count, 2);
    EXPECT_NEAR(bins[0].meanActual, 0.0, 1e-12);
    EXPECT_NEAR(bins[1].meanActual, 1.0, 1e-12);
}

// ---- eval-row schema
// -----------------------------------------------------------------------------

TEST(TrainingDataSchemaTest, EvalHeaderMatchesContractAndNaNRendersAsNan) {
    EXPECT_EQ(std::string(kTrainingEvalHeader),
              "target,model,split,n_train,n_test,auc,log_loss,rmse,calibration_slope,"
              "calibration_intercept,base_rate");
    EXPECT_EQ(countFields(std::string(kTrainingEvalHeader)), 11);
    EXPECT_EQ(std::string(kCalibrationHeader), "bin,mean_pred,mean_actual,count");

    EvalRow r;
    r.target = "completed";
    r.model = "learned";
    r.split = "temporal";
    r.nTrain = 100;
    r.nTest = 25;
    r.auc = 0.82;
    r.logLoss = 0.4;
    r.rmse = std::numeric_limits<double>::quiet_NaN();
    r.calSlope = std::numeric_limits<double>::quiet_NaN();
    r.calIntercept = std::numeric_limits<double>::quiet_NaN();
    r.baseRate = 0.5;
    const std::string line = formatEvalRow(r);
    EXPECT_EQ(countFields(line), 11);
    EXPECT_NE(line.find("nan"), std::string::npos); // NaN cells render as nan
    EXPECT_NE(line.find("completed,learned,temporal,100,25,"), std::string::npos);
}
