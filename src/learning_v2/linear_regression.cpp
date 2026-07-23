#include "rr/learning_v2/linear_regression.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <nlohmann/json.hpp>

namespace rr::learning_v2 {

void LinearRegression::train(const std::vector<std::vector<double>> &x,
                             const std::vector<double> &y, const SgdHyperparams &hp,
                             std::vector<std::string> featureNames, std::string target) {
    hyperparams_ = hp;
    featureNames_ = std::move(featureNames);
    target_ = std::move(target);
    nTrain_ = static_cast<int>(x.size());

    const std::size_t nFeat = featureNames_.size();
    scaler_.fit(x, nFeat);

    std::vector<std::vector<double>> z(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        z[i] = scaler_.transform(x[i]);
    }

    weights_ = initGaussianWeights(nFeat, hp.seed, hp.initSd);
    bias_ = 0.0;
    if (x.empty()) {
        return;
    }

    Rng shuffleRng = forkRng(hp.seed, kTrainingSplitStream);
    std::vector<std::size_t> idx(x.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }
    const std::size_t batch = std::max(1, hp.batchSize);

    for (int epoch = 0; epoch < hp.epochs; ++epoch) {
        fisherYatesShuffle(idx, shuffleRng);
        for (std::size_t start = 0; start < idx.size(); start += batch) {
            const std::size_t end = std::min(start + batch, idx.size());
            const double m = static_cast<double>(end - start);
            std::vector<double> gradW(nFeat, 0.0);
            double gradB = 0.0;
            for (std::size_t k = start; k < end; ++k) {
                const std::vector<double> &zi = z[idx[k]];
                double pred = bias_;
                for (std::size_t j = 0; j < nFeat; ++j) {
                    pred += weights_[j] * zi[j];
                }
                const double err = pred - y[idx[k]]; // d(0.5*MSE)/d(pred)
                for (std::size_t j = 0; j < nFeat; ++j) {
                    gradW[j] += err * zi[j];
                }
                gradB += err;
            }
            for (std::size_t j = 0; j < nFeat; ++j) {
                const double g = gradW[j] / m + hp.l2 * weights_[j];
                weights_[j] -= hp.learningRate * g;
            }
            bias_ -= hp.learningRate * (gradB / m);
        }
    }
}

double LinearRegression::predict(const std::vector<double> &xRaw) const {
    const std::vector<double> z = scaler_.transform(xRaw);
    double dot = bias_;
    for (std::size_t j = 0; j < weights_.size(); ++j) {
        dot += weights_[j] * z[j];
    }
    return dot;
}

nlohmann::json LinearRegression::toJson() const {
    return nlohmann::json{
        {"model_type", "linear_regression"},
        {"target", target_},
        {"model_schema_version", kModelSchemaVersion},
        {"feature_names", featureNames_},
        {"weights", weights_},
        {"bias", bias_},
        {"scaler_mean", scaler_.mean},
        {"scaler_std", scaler_.std},
        {"hyperparams", nlohmann::json{{"epochs", hyperparams_.epochs},
                                       {"batch_size", hyperparams_.batchSize},
                                       {"learning_rate", hyperparams_.learningRate},
                                       {"l2", hyperparams_.l2},
                                       {"init_sd", hyperparams_.initSd},
                                       {"seed", hyperparams_.seed}}},
        {"n_train", nTrain_}};
}

LinearRegression LinearRegression::fromJson(const nlohmann::json &j) {
    LinearRegression m;
    j.at("feature_names").get_to(m.featureNames_);
    j.at("weights").get_to(m.weights_);
    j.at("bias").get_to(m.bias_);
    j.at("scaler_mean").get_to(m.scaler_.mean);
    j.at("scaler_std").get_to(m.scaler_.std);
    j.at("target").get_to(m.target_);
    if (auto it = j.find("n_train"); it != j.end()) {
        it->get_to(m.nTrain_);
    }
    if (auto it = j.find("hyperparams"); it != j.end()) {
        const auto &h = *it;
        h.at("epochs").get_to(m.hyperparams_.epochs);
        h.at("batch_size").get_to(m.hyperparams_.batchSize);
        h.at("learning_rate").get_to(m.hyperparams_.learningRate);
        h.at("l2").get_to(m.hyperparams_.l2);
        h.at("init_sd").get_to(m.hyperparams_.initSd);
        h.at("seed").get_to(m.hyperparams_.seed);
    }
    return m;
}

} // namespace rr::learning_v2
