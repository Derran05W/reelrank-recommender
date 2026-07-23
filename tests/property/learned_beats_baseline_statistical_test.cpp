// Phase 22 package B — the offline half of Tier-5 acceptance 1 (contracts §5/§7, V2 §4.19-4.22).
//
// On a fabricated schema-conformant log with PLANTED signal (outcomes drawn from a known logistic /
// linear function of three feature columns + noise, retrieval_sources assigned INDEPENDENTLY of the
// outcome), the in-house learners must beat the two FREQUENCY baselines on held-out data. Asserts
// are MECHANISM-vs-BASELINE only, with margins set below the demonstrated operating point — NEVER
// the fine ordering between two learned variants (P21 lesson). Also exercises the honest-SKIP path
// (rare `followed`) and pipeline determinism.

#include "rr/learning_v2/training_data.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../unit/learning_v2_test_fixture.hpp"

using namespace rr::learning_v2;
namespace fs = std::filesystem;

namespace {

EvalRow row(const TargetResult &r, const std::string &model) {
    for (const EvalRow &e : r.evalRows) {
        if (e.model == model) {
            return e;
        }
    }
    ADD_FAILURE() << "missing eval row for model " << model;
    return {};
}

} // namespace

class LearnedVsBaseline : public ::testing::Test {
  protected:
    static fs::path dir_;
    static Dataset ds_;

    static void SetUpTestSuite() {
        dir_ = test::makeTempDir("planted");
        test::PlantedLogParams p;
        p.seed = 20260719;
        p.requests = 1000;
        p.shownPerRequest = 4;
        p.poolOnlyPerRequest = 1;
        p.numUsers = 250;
        p.survey = true;
        test::writePlantedLog(dir_, p);
        ds_ = loadTrainingLog(dir_, /*withSurvey=*/true);
    }
    static void TearDownTestSuite() { fs::remove_all(dir_); }

    static SgdHyperparams hp(uint64_t seed = 123) {
        SgdHyperparams h;
        h.epochs = 80;
        h.batchSize = 32;
        h.learningRate = 0.15;
        h.l2 = 1e-4;
        h.seed = seed;
        return h;
    }

    static void reportBinary(const std::string &target, const TargetResult &r) {
        const EvalRow l = row(r, "learned");
        const EvalRow g = row(r, "global_frequency");
        const EvalRow p = row(r, "per_source_frequency");
        const EvalRow s = row(r, "served_score");
        std::cout << "[AUC] " << target << "  learned=" << l.auc << "  global=" << g.auc
                  << "  per_source=" << p.auc << "  served_score=" << s.auc
                  << "  base_rate=" << l.baseRate << "  (n_test=" << l.nTest << ")\n";
    }
    static void reportLinear(const std::string &target, const TargetResult &r) {
        const EvalRow l = row(r, "learned");
        const EvalRow g = row(r, "global_frequency");
        const EvalRow p = row(r, "per_source_frequency");
        std::cout << "[RMSE] " << target << "  learned=" << l.rmse << "  global=" << g.rmse
                  << "  per_source=" << p.rmse << "  cal_slope=" << l.calSlope
                  << "  (n_test=" << l.nTest << ")\n";
    }
};

fs::path LearnedVsBaseline::dir_;
Dataset LearnedVsBaseline::ds_;

TEST_F(LearnedVsBaseline, BinaryTargetsBeatFrequencyBaselines_UserDisjoint) {
    const Split split = assignSplit(ds_, SplitMode::UserDisjoint);
    for (const std::string &target :
         {"completed", "liked", "shared", "session_exit", "not_interested"}) {
        const TargetResult r =
            trainAndEvaluateTarget(ds_, split, *findTarget(target), hp(), SplitMode::UserDisjoint);
        ASSERT_FALSE(r.skipped) << target << ": " << r.skipReason;
        reportBinary(target, r);
        const EvalRow l = row(r, "learned");
        const EvalRow g = row(r, "global_frequency");
        const EvalRow p = row(r, "per_source_frequency");
        // Global frequency is a constant predictor -> AUC is exactly 0.5.
        EXPECT_NEAR(g.auc, 0.5, 1e-9);
        // Mechanism margins (calibrated below the demonstrated operating point).
        EXPECT_GT(l.auc, g.auc + 0.10);
        EXPECT_GT(l.auc, p.auc + 0.08);
    }
}

TEST_F(LearnedVsBaseline, WatchRatioBeatsFrequencyBaselines_UserDisjoint) {
    const Split split = assignSplit(ds_, SplitMode::UserDisjoint);
    const TargetResult r = trainAndEvaluateTarget(ds_, split, *findTarget("watch_ratio"), hp(),
                                                  SplitMode::UserDisjoint);
    ASSERT_FALSE(r.skipped) << r.skipReason;
    reportLinear("watch_ratio", r);
    const EvalRow l = row(r, "learned");
    const EvalRow g = row(r, "global_frequency");
    const EvalRow p = row(r, "per_source_frequency");
    EXPECT_LT(l.rmse, g.rmse * 0.90); // lower RMSE than predicting the mean
    EXPECT_LT(l.rmse, p.rmse * 0.92);
    EXPECT_GT(l.calSlope, 0.7); // well-calibrated slope near 1
    EXPECT_LT(l.calSlope, 1.3);
}

TEST_F(LearnedVsBaseline, SatisfactionProxyBeatsMean_UserDisjoint) {
    const Split split = assignSplit(ds_, SplitMode::UserDisjoint);
    const TargetResult r = trainAndEvaluateTarget(ds_, split, *findTarget("satisfaction"), hp(),
                                                  SplitMode::UserDisjoint);
    ASSERT_FALSE(r.skipped) << r.skipReason;
    reportLinear("satisfaction", r);
    const EvalRow l = row(r, "learned");
    const EvalRow g = row(r, "global_frequency");
    EXPECT_LT(l.rmse, g.rmse * 0.95);
}

TEST_F(LearnedVsBaseline, CompletedAndWatchRatioAlsoBeatBaselines_Temporal) {
    const Split split = assignSplit(ds_, SplitMode::Temporal);
    const TargetResult c =
        trainAndEvaluateTarget(ds_, split, *findTarget("completed"), hp(), SplitMode::Temporal);
    ASSERT_FALSE(c.skipped) << c.skipReason;
    reportBinary("completed(temporal)", c);
    EXPECT_GT(row(c, "learned").auc, row(c, "global_frequency").auc + 0.10);
    EXPECT_GT(row(c, "learned").auc, row(c, "per_source_frequency").auc + 0.08);

    const TargetResult w =
        trainAndEvaluateTarget(ds_, split, *findTarget("watch_ratio"), hp(), SplitMode::Temporal);
    ASSERT_FALSE(w.skipped) << w.skipReason;
    reportLinear("watch_ratio(temporal)", w);
    EXPECT_LT(row(w, "learned").rmse, row(w, "global_frequency").rmse * 0.90);
}

TEST_F(LearnedVsBaseline, RareFollowedTargetSkipsHonestly) {
    const Split split = assignSplit(ds_, SplitMode::UserDisjoint);
    const TargetResult r =
        trainAndEvaluateTarget(ds_, split, *findTarget("followed"), hp(), SplitMode::UserDisjoint);
    EXPECT_TRUE(r.skipped);
    EXPECT_TRUE(r.evalRows.empty());
    EXPECT_TRUE(r.modelJson.empty());
    EXPECT_NE(r.skipReason.find("positives"), std::string::npos);
    std::cout << "[SKIP] followed -> " << r.skipReason << "\n";
}

TEST_F(LearnedVsBaseline, PipelineDeterministicForFixedSeed) {
    const Split split = assignSplit(ds_, SplitMode::UserDisjoint);
    const TargetResult a = trainAndEvaluateTarget(ds_, split, *findTarget("completed"), hp(7),
                                                  SplitMode::UserDisjoint);
    const TargetResult b = trainAndEvaluateTarget(ds_, split, *findTarget("completed"), hp(7),
                                                  SplitMode::UserDisjoint);
    EXPECT_EQ(a.modelJson, b.modelJson); // identical serialized model
    EXPECT_EQ(row(a, "learned").auc, row(b, "learned").auc);
    EXPECT_EQ(row(a, "learned").logLoss, row(b, "learned").logLoss);
}
