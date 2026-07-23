#include "rr/learning_v2/retrainer.hpp"

#include <cstddef>
#include <string>

#include "rr/infrastructure/random.hpp"
#include "rr/learning_v2/training_data.hpp"

namespace rr::learning_v2 {

namespace {

// Golden ratio odd constant (the SplitMix64 gamma), used purely as a per-version salt multiplier so
// successive versions get well-separated learner seeds. Pinned; changing it re-shuffles the model
// init/shuffle streams (a determinism tripwire), so DO NOT change it.
constexpr std::uint64_t kVersionGamma = 0x9E3779B97F4A7C15ULL;

std::vector<double> toDoubleRow(const std::array<float, kNumFeatures> &f) {
    std::vector<double> x(kNumFeatures);
    for (std::size_t j = 0; j < kNumFeatures; ++j) {
        x[j] = static_cast<double>(f[j]);
    }
    return x;
}

// Train one logistic target over ALL rows (x already built), returning nullopt on the honest-SKIP
// rule (too few positives). `label` extracts the 0/1 target from a row.
template <class LabelFn>
std::optional<LogisticRegression>
trainLogistic(const std::vector<std::vector<double>> &x, const std::vector<ShownFeatureRow> &rows,
              const SgdHyperparams &hp, const std::string &target, LabelFn label) {
    std::vector<double> y;
    y.reserve(rows.size());
    int positives = 0;
    for (const ShownFeatureRow &r : rows) {
        const double v = label(r) ? 1.0 : 0.0;
        positives += (v > 0.5) ? 1 : 0;
        y.push_back(v);
    }
    if (static_cast<int>(x.size()) < kMinPositivesToTrain || positives < kMinPositivesToTrain) {
        return std::nullopt; // no learnable signal (e.g. the ~1.5%-base-rate follow target)
    }
    LogisticRegression m;
    m.train(x, y, hp, featureColumnNames(), target);
    return m;
}

} // namespace

std::uint64_t retrainVersionSeed(std::uint64_t runSeed, int version) {
    return splitmix64(runSeed ^ (kVersionGamma * static_cast<std::uint64_t>(version)));
}

Retrainer::Retrainer(std::uint64_t runSeed, std::uint32_t epochs)
    : runSeed_(runSeed), epochs_(epochs) {}

LearnedModels Retrainer::retrain(const std::vector<ShownFeatureRow> &rows, int version) const {
    LearnedModels bundle;
    bundle.version = version;

    SgdHyperparams hp;
    hp.epochs = static_cast<int>(epochs_);
    hp.seed =
        retrainVersionSeed(runSeed_, version); // per-version salted learner master (contracts §3)

    // Shared design matrix (all rows) for the five all-row targets; satisfaction subsets it.
    std::vector<std::vector<double>> x;
    x.reserve(rows.size());
    for (const ShownFeatureRow &r : rows) {
        x.push_back(toDoubleRow(r.features));
    }

    // watch_ratio (linear) — trained over all rows (kMinPositivesToTrain doubles as the min
    // examples).
    if (static_cast<int>(x.size()) >= kMinPositivesToTrain) {
        std::vector<double> y;
        y.reserve(rows.size());
        for (const ShownFeatureRow &r : rows) {
            y.push_back(static_cast<double>(r.watchRatio));
        }
        LinearRegression m;
        m.train(x, y, hp, featureColumnNames(), "watch_ratio");
        bundle.watch = std::move(m);
    }

    // The four binary logistic targets (share / follow / session_exit / not_interested).
    bundle.share = trainLogistic(x, rows, hp, "shared",
                                 [](const ShownFeatureRow &r) { return r.shared != 0; });
    bundle.follow = trainLogistic(x, rows, hp, "followed",
                                  [](const ShownFeatureRow &r) { return r.followed != 0; });
    bundle.exit = trainLogistic(x, rows, hp, "session_exit",
                                [](const ShownFeatureRow &r) { return r.sessionExit != 0; });
    bundle.regret = trainLogistic(x, rows, hp, "not_interested",
                                  [](const ShownFeatureRow &r) { return r.notInterested != 0; });

    // satisfaction (linear) — only rows with a survey likert (contracts §2: absent survey => term
    // 0).
    {
        std::vector<std::vector<double>> xs;
        std::vector<double> ys;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].likert != 0) {
                xs.push_back(x[i]);
                ys.push_back(static_cast<double>(rows[i].likert));
            }
        }
        if (static_cast<int>(xs.size()) >= kMinPositivesToTrain) {
            LinearRegression m;
            m.train(xs, ys, hp, featureColumnNames(), "satisfaction");
            bundle.satisfaction = std::move(m);
        }
    }

    // Ready to serve learned iff at least one term has a model; otherwise the caller keeps the
    // WeightedRanker fallback (a pathological all-skip retrain never happens above
    // min_training_rows, where watch_ratio + the common binaries always meet the threshold —
    // documented).
    bundle.ready = bundle.watch || bundle.share || bundle.follow || bundle.exit || bundle.regret ||
                   bundle.satisfaction;
    return bundle;
}

} // namespace rr::learning_v2
