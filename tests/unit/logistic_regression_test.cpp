// Phase 22 package B — LogisticRegression unit tests: convergence on hand-checkable separable data,
// deterministic training (same seed => identical model; different seed => different model), and a
// bit-exact JSON serialization round-trip (contracts §5, D6/D21).

#include "rr/learning_v2/logistic_regression.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "learning_v2_test_fixture.hpp"
#include "rr/learning_v2/training_data.hpp"

using namespace rr::learning_v2;

namespace {

SgdHyperparams hp(uint64_t seed) {
    SgdHyperparams h;
    h.epochs = 200;
    h.batchSize = 16;
    h.learningRate = 0.2;
    h.l2 = 1e-4;
    h.seed = seed;
    return h;
}

std::vector<double> point(double f0) {
    std::vector<double> x(kNumFeatures, 0.0);
    x[0] = f0;
    return x;
}

} // namespace

TEST(LogisticRegressionTest, ConvergesOnSeparableData) {
    const test::XY tr = test::makeSeparableBinary(400, 1);
    LogisticRegression m;
    m.train(tr.x, tr.y, hp(1), featureColumnNames(), "completed");

    // The decision must follow feature 0 with a positive weight and clean separation on held-out.
    EXPECT_GT(m.weights()[0], 0.5);
    EXPECT_GT(m.predictProba(point(0.9)), 0.5);
    EXPECT_LT(m.predictProba(point(0.1)), 0.5);
    EXPECT_GT(m.predictProba(point(0.9)), m.predictProba(point(0.1)));

    const test::XY te = test::makeSeparableBinary(200, 2);
    std::vector<double> preds;
    preds.reserve(te.x.size());
    for (const auto &x : te.x) {
        preds.push_back(m.predictProba(x));
    }
    EXPECT_GT(rankAuc(preds, te.y), 0.98);
}

TEST(LogisticRegressionTest, SameSeedIdenticalDifferentSeedDiffers) {
    const test::XY tr = test::makeSeparableBinary(300, 5);
    LogisticRegression a;
    a.train(tr.x, tr.y, hp(42), featureColumnNames(), "completed");
    LogisticRegression b;
    b.train(tr.x, tr.y, hp(42), featureColumnNames(), "completed");
    LogisticRegression c;
    c.train(tr.x, tr.y, hp(43), featureColumnNames(), "completed");

    EXPECT_EQ(a.weights(), b.weights()); // same seed => bit-identical
    EXPECT_EQ(a.bias(), b.bias());
    EXPECT_NE(a.weights(), c.weights()); // different seed => different model
}

TEST(LogisticRegressionTest, JsonRoundTripBitExact) {
    const test::XY tr = test::makeSeparableBinary(250, 9);
    LogisticRegression m;
    m.train(tr.x, tr.y, hp(7), featureColumnNames(), "liked");

    // Round-trip THROUGH the serialized text (what the app writes/reads).
    const nlohmann::json j = nlohmann::json::parse(m.toJson().dump());
    const LogisticRegression reloaded = LogisticRegression::fromJson(j);

    EXPECT_EQ(m.weights(), reloaded.weights());
    EXPECT_EQ(m.bias(), reloaded.bias());
    EXPECT_EQ(m.target(), reloaded.target());
    for (double f0 : {0.05, 0.25, 0.5, 0.75, 0.95}) {
        EXPECT_EQ(m.predictProba(point(f0)), reloaded.predictProba(point(f0)));
    }
}

TEST(LogisticRegressionTest, L2RegularizationShrinksWeightNorm) {
    const test::XY tr = test::makeSeparableBinary(300, 3);
    auto norm = [](const LogisticRegression &m) {
        double s = 0.0;
        for (double w : m.weights()) {
            s += w * w;
        }
        return std::sqrt(s);
    };
    SgdHyperparams weak = hp(11);
    weak.l2 = 1e-5;
    SgdHyperparams strong = hp(11);
    strong.l2 = 1.0;
    LogisticRegression a;
    a.train(tr.x, tr.y, weak, featureColumnNames(), "completed");
    LogisticRegression b;
    b.train(tr.x, tr.y, strong, featureColumnNames(), "completed");
    EXPECT_LT(norm(b), norm(a));
}
