#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/infrastructure/random.hpp"

// ================================================================================================
// Phase 22 package B — shared deterministic-SGD primitives for the in-house learners (D21).
//
// Both LogisticRegression and LinearRegression use identical mini-batch SGD plumbing: z-score
// feature standardization fit on the TRAIN split only, "model-init"-forked gaussian weight init,
// and a per-epoch Fisher-Yates shuffle driven by a "training-split"-forked rr::Rng. Those pieces
// live here so the two learners cannot drift apart and so training_data's offline-eval harness can
// depend on the primitives without depending on either learner (a clean acyclic include graph:
// sgd_common <- {logistic,linear,training_data}; {logistic,linear} <- training_data).
//
// D8/D19 RNG discipline: randomness enters ONLY through rr::forkRng on the two pinned stream names
// "model-init" and "training-split", forked from the CLI master seed. No std::*_distribution is
// ever used (banned, D8). Header-only/inline so no extra TU is added to rr_learning_v2.
// ================================================================================================

namespace rr::learning_v2 {

// The training-log feature-column count (rr::FeatureVector, mirrored by kFeatureColumns in
// training_log_schema.hpp). The learners operate on fixed-width rows of this many features.
inline constexpr std::size_t kNumFeatures = 21;

// Version stamp for the model-JSON format (distinct from the training-log kSchemaVersion). Bump if
// the serialized model key set changes; fromJson tolerates older/newer minor content but records
// it. Kept simple: a single integer echoed into every model file.
inline constexpr int kModelSchemaVersion = 1;

// The two D19-pinned rr::Rng stream names this phase forks (00-DESIGN-DECISIONS-V2.md D19).
inline constexpr std::string_view kModelInitStream = "model-init";
inline constexpr std::string_view kTrainingSplitStream = "training-split";

// Numerically-stable logistic sigmoid.
inline double sigmoid(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    const double e = std::exp(z);
    return e / (1.0 + e);
}

// Mini-batch SGD hyperparameters. Defaults are documented at their definition (D24 convention) and
// are the values the convergence/statistical tests are calibrated at; the CLI exposes epochs,
// batch size, learning rate and L2 as optional overrides (contracts §5 requires only the core
// flags, so these carry sensible defaults).
struct SgdHyperparams {
    // Full passes over the train split. 200 comfortably converges the standardized-feature linear
    // models on the fixture scale (a few thousand rows) in well under a second.
    int epochs = 200;
    // Mini-batch size; the gradient is AVERAGED over the batch. 32 is the classic default.
    int batchSize = 32;
    // Fixed (non-decaying) step size. With z-scored features the loss surface is well conditioned,
    // so a constant 0.1 is stable and hand-checkable; documented rather than tuned per D24.
    double learningRate = 0.1;
    // L2 penalty on the WEIGHTS only (never the bias). Small by default so it regularizes without
    // materially shrinking a strong planted signal; exposed on the CLI (--l2).
    double l2 = 1e-4;
    // Std-dev of the "model-init"-forked gaussian weight initialization (bias starts at 0).
    double initSd = 0.01;
    // Master seed; the learner forks "model-init" and "training-split" from it.
    uint64_t seed = 0;
};

// Per-feature z-score standardizer, FIT ON THE TRAIN SPLIT ONLY and then stored in the model JSON
// (contracts §5). transform maps x -> (x - mean) / std. A feature with (near-)zero variance on the
// train split gets std = 1 so its centered value is ~0 and it simply contributes nothing — this is
// how the all-zero V2 feature columns (gate-off logs) fall out harmlessly. Documented behaviour.
struct StandardScaler {
    std::vector<double> mean;
    std::vector<double> std;

    // Fit population mean and std over rows x[i] (each of width nFeatures). Empty input yields
    // identity scaling (mean 0, std 1) for the fixed feature width.
    void fit(const std::vector<std::vector<double>> &x, std::size_t nFeatures) {
        mean.assign(nFeatures, 0.0);
        std.assign(nFeatures, 1.0);
        if (x.empty()) {
            return;
        }
        const double n = static_cast<double>(x.size());
        for (const auto &row : x) {
            for (std::size_t j = 0; j < nFeatures; ++j) {
                mean[j] += row[j];
            }
        }
        for (std::size_t j = 0; j < nFeatures; ++j) {
            mean[j] /= n;
        }
        std::vector<double> var(nFeatures, 0.0);
        for (const auto &row : x) {
            for (std::size_t j = 0; j < nFeatures; ++j) {
                const double d = row[j] - mean[j];
                var[j] += d * d;
            }
        }
        for (std::size_t j = 0; j < nFeatures; ++j) {
            const double sd = std::sqrt(var[j] / n);
            std[j] = (sd < 1e-12) ? 1.0 : sd; // constant feature -> std 1 -> centered value ~0
        }
    }

    std::vector<double> transform(const std::vector<double> &row) const {
        std::vector<double> z(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            z[j] = (row[j] - mean[j]) / std[j];
        }
        return z;
    }
};

// "model-init"-forked small-gaussian weight vector (bias handled separately by the caller, starts
// at 0). Using gaussians (not zeros) is a deliberate choice so the pinned "model-init" stream is
// actually consumed (D19) and two seeds yield distinguishable models — the split-determinism test
// asserts identical init for identical seeds and differing init for differing seeds.
inline std::vector<double> initGaussianWeights(std::size_t n, uint64_t seed, double sd) {
    Rng rng = forkRng(seed, kModelInitStream);
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = sd * rng.gaussian();
    }
    return w;
}

// In-place Fisher-Yates shuffle using an already-forked "training-split" rr::Rng, so successive
// epochs draw a fresh permutation from the SAME advancing stream (fixed, seed-determined order).
inline void fisherYatesShuffle(std::vector<std::size_t> &idx, Rng &rng) {
    for (std::size_t i = idx.size(); i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(rng.uniformInt(i)); // [0, i)
        std::swap(idx[i - 1], idx[j]);
    }
}

} // namespace rr::learning_v2
