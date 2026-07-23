#include "rr/learning_v2/learned_ranker.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include <nlohmann/json.hpp>

#include "rr/learning_v2/sgd_common.hpp"
#include "rr/learning_v2/training_log_schema.hpp"

namespace rr::learning_v2 {

namespace {

// Flatten a FeatureVector into the length-kNumFeatures raw feature row the models were TRAINED on,
// in kFeatureColumns / FeatureVector DECLARATION order (contracts §2). This ordering is the frozen
// serving-purity contract: it is byte-for-byte the same sequence TrainingLogger::writeFeatures
// emits to candidates.csv and training_data reads back, so the vector this ranker scores == the
// logged row for the same (request, reel). If FeatureVector ever changes, kFeatureColumns AND this
// function AND writeFeatures must move together (all three are asserted against kNumFeatures==21).
std::vector<double> featuresToRow(const FeatureVector &f) {
    static_assert(kNumFeatures == 21, "featuresToRow mirrors the 21 kFeatureColumns");
    return {
        f.similarity,         f.sessionTopic,    f.quality,
        f.freshness,          f.popularity,      f.trending,
        f.creatorAffinity,    f.exploration,     f.durationMatch,
        f.repetition,         f.impressionCount, f.visualMatch,
        f.musicMatch,         f.emotionalMatch,  f.clickbait,
        f.emotionalIntensity, f.usefulness,      f.productionQuality,
        f.informationDensity, f.languageMatch,   f.savePopularity,
    };
}

double clampd(double v, double lo, double hi) { return std::clamp(v, lo, hi); }

} // namespace

std::string LearnedModels::trainedTargets() const {
    // Canonical order matches the value-function term order; absent (skipped) targets are omitted.
    std::string out;
    auto add = [&](bool present, const char *name) {
        if (!present) {
            return;
        }
        if (!out.empty()) {
            out += '|';
        }
        out += name;
    };
    add(watch.has_value(), "watch_ratio");
    add(share.has_value(), "shared");
    add(follow.has_value(), "followed");
    add(exit.has_value(), "session_exit");
    add(regret.has_value(), "not_interested");
    add(satisfaction.has_value(), "satisfaction");
    return out;
}

nlohmann::json LearnedModels::toJson() const {
    nlohmann::json j;
    j["version"] = version;
    j["ready"] = ready;
    nlohmann::json models = nlohmann::json::object();
    if (watch) {
        models["watch_ratio"] = watch->toJson();
    }
    if (share) {
        models["shared"] = share->toJson();
    }
    if (follow) {
        models["followed"] = follow->toJson();
    }
    if (exit) {
        models["session_exit"] = exit->toJson();
    }
    if (regret) {
        models["not_interested"] = regret->toJson();
    }
    if (satisfaction) {
        models["satisfaction"] = satisfaction->toJson();
    }
    j["models"] = std::move(models);
    return j;
}

LearnedRanker::LearnedRanker(const std::vector<Reel> &reels, const RankingConfig &ranking,
                             bool contentV2, bool personalizedDiversity,
                             const LearningV2ValueWeights &weights)
    : extractor_(reels, ranking, contentV2),
      fallbackRanker_(reels, ranking, contentV2, personalizedDiversity), weights_(weights) {}

std::vector<Candidate> LearnedRanker::rank(const User &user,
                                           const std::vector<Candidate> &candidates,
                                           Timestamp now) const {
    ++rankCalls_;

    // Cold-start fallback (contracts §2/§5): below the data threshold no ready models exist, so
    // serve the hand-tuned WeightedRanker scores EXACTLY (delegating to the owned,
    // identically-constructed ranker) and mark fallback=1 on every candidate. The scores/order are
    // byte-identical to a standalone WeightedRanker; only the additive fallback key distinguishes
    // the map.
    if (!models_.ready) {
        ++fallbackCalls_;
        std::vector<Candidate> out = fallbackRanker_.rank(user, candidates, now);
        for (Candidate &c : out) {
            c.featureContributions["fallback"] = 1.0F;
        }
        return out;
    }

    // Learned mode: score by the §4.21 multi-objective value. Extract the served-time features with
    // the SAME extractor the log uses (serving purity), map each to the trained feature row, and
    // combine the six predicted terms.
    const std::vector<FeatureVector> features = extractor_.extract(user, candidates, now);

    std::vector<Candidate> ranked = candidates;
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        const std::vector<double> x = featuresToRow(features[i]);

        // §2 term mappings (documented at definition):
        //  pWatch          = watch_ratio linear, clamped [0,1.5] then /1.5-normalized to [0,1].
        //  pShare/pFollow/pExit/pRegret = the logistic proba (already [0,1]); pExit is the
        //                    observed_exit_after_impression model, pRegret the not_interested model
        //                    (the frozen designed observable regret proxy, contracts §2).
        //  pSatisfaction   = survey linear (Likert 1..5) mapped (pred−1)/4 clamped to [0,1]; when
        //  the
        //                    survey model is absent the term is 0 and satisfaction_available=0.
        //  A binary target absent (P22 honest-SKIP) contributes 0 for that term.
        const double pWatch =
            models_.watch ? clampd(models_.watch->predict(x), 0.0, 1.5) / 1.5 : 0.0;
        const double pShare = models_.share ? models_.share->predictProba(x) : 0.0;
        const double pFollow = models_.follow ? models_.follow->predictProba(x) : 0.0;
        const double pExit = models_.exit ? models_.exit->predictProba(x) : 0.0;
        const double pRegret = models_.regret ? models_.regret->predictProba(x) : 0.0;
        const bool satAvailable = models_.satisfaction.has_value();
        const double pSatisfaction =
            satAvailable ? clampd((models_.satisfaction->predict(x) - 1.0) / 4.0, 0.0, 1.0) : 0.0;

        // Weighted terms (V1 §14.4 explanation parity). exit/regret enter NEGATIVELY, mirroring the
        // WeightedRanker penalty convention, so the explanation map's values sum directly to the
        // learned value. Computed in double, stored as float; learned_value is the float sum of the
        // stored float terms (== Σ terms to float tolerance, property-tested).
        const float predictedWatch = static_cast<float>(weights_.watch * pWatch);
        const float predictedShare = static_cast<float>(weights_.share * pShare);
        const float predictedFollow = static_cast<float>(weights_.follow * pFollow);
        const float predictedSatisfaction =
            static_cast<float>(weights_.satisfaction * pSatisfaction);
        const float predictedExit = static_cast<float>(-(weights_.exit * pExit));
        const float predictedRegret = static_cast<float>(-(weights_.regret * pRegret));

        const double value = static_cast<double>(predictedWatch) + predictedShare +
                             predictedFollow + predictedSatisfaction + predictedExit +
                             predictedRegret;

        Candidate &c = ranked[i];
        c.rankingScore = static_cast<float>(value);
        c.featureContributions = {
            {"predicted_watch", predictedWatch},
            {"predicted_share", predictedShare},
            {"predicted_follow", predictedFollow},
            {"predicted_satisfaction", predictedSatisfaction},
            {"predicted_exit", predictedExit},
            {"predicted_regret", predictedRegret},
            {"learned_value", static_cast<float>(value)},
            {"fallback", 0.0F},
            {"satisfaction_available", satAvailable ? 1.0F : 0.0F},
        };
    }

    // Same total order as the WeightedRanker / Orchestrator: score DESCENDING, ties by ascending
    // ReelId (unique in a deduplicated pool => fully deterministic).
    std::sort(ranked.begin(), ranked.end(), [](const Candidate &a, const Candidate &b) {
        if (a.rankingScore != b.rankingScore) {
            return a.rankingScore > b.rankingScore;
        }
        return a.reelId.value < b.reelId.value;
    });
    return ranked;
}

void LearnedRanker::setModels(LearnedModels models) { models_ = std::move(models); }

} // namespace rr::learning_v2
