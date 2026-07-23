#pragma once

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "rr/learning_v2/sgd_common.hpp"

// ================================================================================================
// Phase 22 package B — in-house LOGISTIC REGRESSION for the binary §4.19 targets (completed / liked
// / shared / followed / not_interested / session_exit). Deterministic mini-batch SGD on the
// log-loss with an optional L2 penalty (D21). Feature standardization is fit on the train split and
// stored in the model so predict() is self-contained; JSON round-trips bit-exactly (D6).
// ================================================================================================

namespace rr::learning_v2 {

class LogisticRegression {
  public:
    // Fit on the TRAIN rows only. x[i] is a length-nFeatures raw feature row, y[i] in {0,1}.
    // featureNames (length nFeatures) and target are stored for the model JSON / self-description.
    void train(const std::vector<std::vector<double>> &x, const std::vector<double> &y,
               const SgdHyperparams &hp, std::vector<std::string> featureNames, std::string target);

    // P(y=1 | xRaw): sigmoid(w . standardize(xRaw) + b). Deterministic; safe before/after JSON
    // reload (identical doubles => identical result).
    double predictProba(const std::vector<double> &xRaw) const;

    // D6 JSON serialization with bit-exact predict round-trip (nlohmann's shortest-round-trip
    // double formatting guarantees the reloaded weights/scaler are the identical doubles).
    nlohmann::json toJson() const;
    static LogisticRegression fromJson(const nlohmann::json &j);

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
