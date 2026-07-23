#pragma once

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "rr/learning_v2/sgd_common.hpp"

// ================================================================================================
// Phase 22 package B — in-house LINEAR REGRESSION for the continuous §4.19 targets (watch_ratio and
// the survey satisfaction proxy). Deterministic mini-batch SGD on mean-squared error with optional
// L2 (D21), sharing the sgd_common standardization / init / shuffle plumbing with the logistic
// learner. Predictions are the raw linear response (unclamped — documented; the calibration report
// and RMSE handle any excursion outside the target's natural range). JSON round-trips exactly (D6).
// ================================================================================================

namespace rr::learning_v2 {

class LinearRegression {
  public:
    void train(const std::vector<std::vector<double>> &x, const std::vector<double> &y,
               const SgdHyperparams &hp, std::vector<std::string> featureNames, std::string target);

    // Raw linear response w . standardize(xRaw) + b (not clamped). Deterministic and stable across
    // a JSON round-trip.
    double predict(const std::vector<double> &xRaw) const;

    nlohmann::json toJson() const;
    static LinearRegression fromJson(const nlohmann::json &j);

    const std::vector<double> &weights() const { return weights_; }
    double bias() const { return bias_; }
    const StandardScaler &scaler() const { return scaler_; }
    const std::vector<std::string> &featureNames() const { return featureNames_; }
    const std::string &target() const { return target_; }
    int nTrain() const { return nTrain_; }

  private:
    std::vector<double> weights_;
    double bias_ = 0.0;
    StandardScaler scaler_;
    std::vector<std::string> featureNames_;
    std::string target_;
    SgdHyperparams hyperparams_;
    int nTrain_ = 0;
};

} // namespace rr::learning_v2
