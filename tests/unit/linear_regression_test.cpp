// Phase 22 package B — LinearRegression unit tests: convergence on a hand-checkable linear target,
// deterministic training, and a bit-exact JSON serialization round-trip (contracts §5, D6/D21).

#include "rr/learning_v2/linear_regression.hpp"

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
    h.epochs = 300;
    h.batchSize = 16;
    h.learningRate = 0.1;
    h.l2 = 1e-5;
    h.seed = seed;
    return h;
}

std::vector<double> point(double f0, double f1) {
    std::vector<double> x(kNumFeatures, 0.0);
    x[0] = f0;
    x[1] = f1;
    return x;
}

} // namespace

TEST(LinearRegressionTest, ConvergesOnLinearTarget) {
    // y = 0.3 + 1.5*f0 - 0.8*f1 (+ tiny noise).
    const test::XY tr = test::makeLinearTarget(500, 1);
    LinearRegression m;
    m.train(tr.x, tr.y, hp(1), featureColumnNames(), "watch_ratio");

    const test::XY te = test::makeLinearTarget(300, 2);
    std::vector<double> preds;
    std::vector<double> actual = te.y;
    preds.reserve(te.x.size());
    for (const auto &x : te.x) {
        preds.push_back(m.predict(x));
    }
    EXPECT_LT(rmse(preds, actual), 0.05);

    const CalibrationFit cal = calibrationFit(preds, actual);
    EXPECT_NEAR(cal.slope, 1.0, 0.1);

    // Monotone in each signal feature with the correct sign.
    EXPECT_GT(m.predict(point(0.9, 0.5)), m.predict(point(0.1, 0.5)));
    EXPECT_LT(m.predict(point(0.5, 0.9)), m.predict(point(0.5, 0.1)));
}

TEST(LinearRegressionTest, SameSeedIdenticalDifferentSeedDiffers) {
    const test::XY tr = test::makeLinearTarget(300, 5);
    LinearRegression a;
    a.train(tr.x, tr.y, hp(42), featureColumnNames(), "watch_ratio");
    LinearRegression b;
    b.train(tr.x, tr.y, hp(42), featureColumnNames(), "watch_ratio");
    LinearRegression c;
    c.train(tr.x, tr.y, hp(43), featureColumnNames(), "watch_ratio");
    EXPECT_EQ(a.weights(), b.weights());
    EXPECT_EQ(a.bias(), b.bias());
    EXPECT_NE(a.weights(), c.weights());
}

TEST(LinearRegressionTest, JsonRoundTripBitExact) {
    const test::XY tr = test::makeLinearTarget(250, 9);
    LinearRegression m;
    m.train(tr.x, tr.y, hp(7), featureColumnNames(), "watch_ratio");

    const nlohmann::json j = nlohmann::json::parse(m.toJson().dump());
    const LinearRegression reloaded = LinearRegression::fromJson(j);

    EXPECT_EQ(m.weights(), reloaded.weights());
    EXPECT_EQ(m.bias(), reloaded.bias());
    for (double f0 : {0.05, 0.5, 0.95}) {
        for (double f1 : {0.1, 0.9}) {
            EXPECT_EQ(m.predict(point(f0, f1)), reloaded.predict(point(f0, f1)));
        }
    }
}
