#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "rr/domain/candidate.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/learning_v2/linear_regression.hpp"
#include "rr/learning_v2/logistic_regression.hpp"
#include "rr/recommendation/feature_extractor.hpp"
#include "rr/recommendation/ranker.hpp"
#include "rr/recommendation/weighted_ranker.hpp"

// ================================================================================================
// Phase 23 LEARNED multi-objective ranker (contracts docs/design/P23-CONTRACTS.md §2, V2 TDD
// §4.21). A Ranker that scores each candidate by the §4.21 multi-objective value from the P22
// in-house models (LogisticRegression / LinearRegression), retrained in-loop by the Retrainer and
// hot-swapped in between requests. Lives in rr_learning_v2 (which depends on rr_recommend) so it
// can OWN a WeightedRanker for the cold-start fallback and reuse the exact same FeatureExtractor
// the serving pipeline logs — serving purity by construction (V2 §10 item 8): the FeatureVector
// this ranker scores IS the one the TrainingLogger records for that (request, reel).
//
// D18: this header pulls in ONLY recommender-visible surfaces (feature_extractor / weighted_ranker
// / the learners) — never simulation/hidden/, so the D18 include-graph guard (which scans
// include/rr/learning_v2) passes. The models are trained on observable features + observable
// outcome labels + the sanctioned survey likert only.
// ================================================================================================

namespace rr::learning_v2 {

// One retrained bundle of the six §4.21 value-function models (contracts §2/§3). A target whose
// signal was too sparse to train (P22's honest-SKIP rule; e.g. the ~1.5%-base-rate follow, or
// satisfaction when the survey is off) is left absent — its predicted term then contributes 0
// (satisfaction additionally sets satisfaction_available=0 in the explanation). `ready` is false
// for the empty cold-start bundle (version 0), which routes rank() to the WeightedRanker fallback.
struct LearnedModels {
    int version = 0;    // 0 = cold-start (no models); >=1 = the retrain sequence number
    bool ready = false; // false => LearnedRanker serves the WeightedRanker fallback (fallback=1)

    std::optional<LinearRegression> watch;    // pWatch  — watch_ratio linear
    std::optional<LogisticRegression> share;  // pShare  — shared logistic
    std::optional<LogisticRegression> follow; // pFollow — followed logistic
    std::optional<LogisticRegression> exit;   // pExit   — observed_exit_after_impression logistic
    std::optional<LogisticRegression> regret; // pRegret — not_interested logistic (§2 frozen)
    std::optional<LinearRegression> satisfaction; // pSatisfaction — survey satisfaction linear

    // Pipe-delimited list of the targets actually trained this version, in canonical order (the
    // retraining_log.csv `targets_trained` cell). Absent targets are omitted.
    std::string trainedTargets() const;

    // Deterministic, bit-exact JSON dump of the whole bundle (version + every present model via the
    // learners' own round-trip toJson). Two same-seed runs produce byte-identical dumps — the
    // retraining-determinism test compares these (contracts §5). Not written to disk by default.
    nlohmann::json toJson() const;
};

class LearnedRanker final : public Ranker {
  public:
    // Constructed with the SAME (reels, ranking config, contentV2, personalizedDiversity) every
    // ranked recommender hands its own FeatureExtractor + WeightedRanker, so (a) the fallback
    // scores are byte-identical to a standalone WeightedRanker (cold-start exactness, contracts §5)
    // and (b) the learned-mode feature vectors are byte-identical to the logged ones (serving
    // purity). `weights` are the six §4.21 value-function weights (config
    // learning_v2.value_weights).
    LearnedRanker(const std::vector<Reel> &reels, const RankingConfig &ranking, bool contentV2,
                  bool personalizedDiversity, const LearningV2ValueWeights &weights);

    // Ranker interface. When no ready models are present (cold start) delegates to the owned
    // WeightedRanker EXACTLY and marks fallback=1 on every candidate; otherwise scores each
    // candidate by the §4.21 value and fills the §2 explanation map. Sorted descending by score,
    // ties by ascending ReelId (the WeightedRanker/Orchestrator total order). Deterministic; no
    // rng.
    std::vector<Candidate> rank(const User &user, const std::vector<Candidate> &candidates,
                                Timestamp now) const override;

    // Hot-swap the model bundle (contracts §3). Called by the runner BETWEEN requests, at the
    // RequestFeed retrain hook before recommend(), so the swap point is deterministic and the
    // triggering request is served by the freshly-swapped models. Non-const (mutates state).
    void setModels(LearnedModels models);

    bool hasModels() const { return models_.ready; }
    int version() const { return models_.version; }
    const LearnedModels &models() const { return models_; }

    // Fallback accounting for the `learned_models` summary block (contracts §3): the share of
    // rank() calls (one per feed request) served by the cold-start fallback. Counters are mutable
    // because rank() is const; single-threaded, so no synchronization is needed.
    std::uint64_t rankCalls() const { return rankCalls_; }
    std::uint64_t fallbackCalls() const { return fallbackCalls_; }

  private:
    FeatureExtractor extractor_;    // learned-mode feature extraction (same config as the log's)
    WeightedRanker fallbackRanker_; // cold-start delegate (hand-tuned scores, byte-exact)
    LearningV2ValueWeights weights_;
    LearnedModels models_; // swapped in by the runner; the empty default routes to fallback

    mutable std::uint64_t rankCalls_ = 0;
    mutable std::uint64_t fallbackCalls_ = 0;
};

} // namespace rr::learning_v2
