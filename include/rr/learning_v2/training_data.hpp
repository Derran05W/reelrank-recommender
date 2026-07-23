#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rr/learning_v2/sgd_common.hpp"

// ================================================================================================
// Phase 22 package B — training-data reader, deterministic splits, target extraction, and the
// offline-evaluation harness (contracts docs/design/P22-CONTRACTS.md §5). Reads the FROZEN §2 CSV
// tables (schema names in training_log_schema.hpp), joins candidates × outcomes on
// (request_id, reel_id) keeping SHOWN rows only, attaches request metadata + the optional survey,
// assigns the temporal / user-disjoint splits, extracts the eight §4.19 targets, and computes the
// AUC / log-loss / RMSE / calibration metrics and the three baselines. This module owns everything
// testable so apps/train_models.cpp stays a thin CLI wrapper.
//
// TOLERANCES (integration-facing — flagged in the report): CSV is parsed by HEADER NAME (extra and
// reordered columns are fine; a missing REQUIRED column throws); rotation part files matching
// "<base>-partNNNN.csv" are globbed and concatenated; fields are simple comma-split (no embedded
// commas / quoting — retrieval_sources must use a non-comma delimiter, which A's writer does);
// empty / unparseable numeric cells become 0.0.
// ================================================================================================

namespace rr::learning_v2 {

// MY frozen SPLIT salt for user-disjoint assignment (contracts §5: "a frozen split salt in YOUR
// header, distinct from the two sampling salts"). ASCII "REELSPLT" — distinct from kLogSampleSalt
// ("REELSHWN") and kLogPoolSampleSalt ("REELPOOL") in training_log_schema.hpp. Changing it
// re-shuffles user membership, so it is pinned like the sampling salts.
inline constexpr uint64_t kSplitSalt = 0x5245454C53504C54ULL; // "REELSPLT"

// First 80% of requests (temporal) / users with hash < 0.8 (user-disjoint) are the TRAIN split.
inline constexpr double kTrainFraction = 0.8;

// Honest-SKIP threshold (contracts §5 / §7): a target with fewer than this many positives (binary)
// or examples (linear) in the train split is skipped with a message rather than trained — the
// rare-follow / sparse-survey path that activates at integration scale.
inline constexpr int kMinPositivesToTrain = 20;

// ---- one joined, SHOWN training example --------------------------------------------------------
struct Example {
    uint64_t requestId = 0;
    uint64_t reelId = 0;
    uint64_t userId = 0;
    uint64_t timestamp = 0; // request timestamp (from requests.csv), used by the temporal split

    std::array<double, kNumFeatures> features{}; // in kFeatureColumns (schema) order
    double servedScore = 0.0;                    // served-score baseline predictor
    std::string retrievalSources;                // majority-label per-source baseline key

    // Outcomes trained/evaluated against (the six binary + watch_ratio).
    double watchRatio = 0.0;
    double completed = 0.0;
    double liked = 0.0;
    double shared = 0.0;
    double followed = 0.0;
    double notInterested = 0.0;
    double sessionExit = 0.0; // observed_exit_after_impression

    // Survey satisfaction proxy (only when survey.csv joins this (request_id, reel_id)).
    bool hasSurvey = false;
    double likert = 0.0; // 1..5
};

struct Dataset {
    std::vector<Example> rows;
    // Diagnostics filled by loadTrainingLog (reported by the CLI, asserted by tests).
    std::size_t shownCandidates = 0;   // shown rows read from candidates*.csv
    std::size_t joinedWithOutcome = 0; // shown rows that joined an outcome row
    std::size_t droppedNoRequest = 0;  // shown rows whose request_id was absent from requests.csv
    std::size_t surveyRows = 0;        // rows that joined a survey likert
};

// Load + join the §2 tables under `dir` (see file header for tolerances). withSurvey joins
// survey*.csv when present (absent => no survey rows, satisfaction target later skips). Throws
// std::runtime_error on a missing required table/column.
Dataset loadTrainingLog(const std::filesystem::path &dir, bool withSurvey);

// The 21 feature column names in schema order (from kFeatureColumns) — passed to the learners and
// echoed into every model JSON.
const std::vector<std::string> &featureColumnNames();

// Feature row in schema order (length kNumFeatures).
std::vector<double> featuresOf(const Example &e);

// ---- targets -----------------------------------------------------------------------------------
enum class TargetKind { Binary, Linear };

struct TargetSpec {
    std::string name;    // model-<name>.json + training_eval `target`
    TargetKind kind;     // Binary => logistic, Linear => linear regression
    bool requiresSurvey; // true only for "satisfaction"
};

// The eight §4.19 targets in canonical order: six binary, watch_ratio, satisfaction.
const std::vector<TargetSpec> &allTargets();
const TargetSpec *findTarget(std::string_view name); // nullptr if unknown

// Target value for an example, or nullopt when inapplicable (satisfaction on a non-survey row).
std::optional<double> extractTarget(const Example &e, const TargetSpec &t);

// ---- splits ------------------------------------------------------------------------------------
enum class SplitMode { Temporal, UserDisjoint };
SplitMode parseSplitMode(std::string_view s); // throws std::invalid_argument on a bad value
std::string_view splitModeName(SplitMode m);

struct Split {
    std::vector<std::size_t> train; // indices into Dataset::rows
    std::vector<std::size_t> test;
};
Split assignSplit(const Dataset &ds, SplitMode mode);

// Deterministic per-source baseline key: the majority single-source token of a retrieval_sources
// field, splitting on | + ; / and whitespace; ties break lexicographically; "" => "unknown".
std::string majoritySourceKey(const std::string &retrievalSources);

// ---- metrics -----------------------------------------------------------------------------------
// Rank-based AUC with tied predictions averaged (deterministic). NaN if a class is absent.
double rankAuc(const std::vector<double> &pred, const std::vector<double> &label);
// Binary cross-entropy with p clamped to [1e-6, 1-1e-6].
double logLoss(const std::vector<double> &pred, const std::vector<double> &label);
double rmse(const std::vector<double> &pred, const std::vector<double> &actual);

struct CalibrationFit {
    double slope;
    double intercept;
};
// Least-squares actual ~ slope*pred + intercept; {NaN,NaN} if pred has ~zero variance.
CalibrationFit calibrationFit(const std::vector<double> &pred, const std::vector<double> &actual);

struct CalibrationBin {
    int bin;
    double meanPred;
    double meanActual;
    int count;
};
// Sort by prediction, partition into <=nbins equal-count deciles; empty bins omitted.
std::vector<CalibrationBin> equalCountBins(const std::vector<double> &pred,
                                           const std::vector<double> &actual, int nbins);

// ---- eval-row schema (contracts §5, frozen columns) --------------------------------------------
struct EvalRow {
    std::string target;
    std::string model; // "learned" | "global_frequency" | "per_source_frequency" | "served_score"
    std::string split; // "temporal" | "user_disjoint"
    int nTrain = 0;
    int nTest = 0;
    double auc = 0.0;
    double logLoss = 0.0;
    double rmse = 0.0;
    double calSlope = 0.0;
    double calIntercept = 0.0;
    double baseRate = 0.0;
};

// The frozen training_eval.csv header and calibration-<target>.csv header (contracts §5).
inline constexpr std::string_view kTrainingEvalHeader =
    "target,model,split,n_train,n_test,auc,log_loss,rmse,calibration_slope,calibration_intercept,"
    "base_rate";
inline constexpr std::string_view kCalibrationHeader = "bin,mean_pred,mean_actual,count";

std::string formatEvalRow(const EvalRow &r);
std::string formatCalibrationBin(const CalibrationBin &b);

// ---- per-target train + evaluate (the orchestration shared by the CLI and the tests) -----------
struct TargetResult {
    bool skipped = false;
    std::string skipReason;
    std::string modelJson;                   // dumped model-<target>.json (empty when skipped)
    std::vector<EvalRow> evalRows;           // learned + the three baselines (empty when skipped)
    std::vector<CalibrationBin> calibration; // learned-model deciles on test (empty when skipped)
};

// Train the learned model for one target and evaluate it + the three baselines on the held-out
// test split. Applies the honest-SKIP rule (returns skipped=true with a reason). Deterministic in
// `hp.seed`.
TargetResult trainAndEvaluateTarget(const Dataset &ds, const Split &split, const TargetSpec &target,
                                    const SgdHyperparams &hp, SplitMode mode);

} // namespace rr::learning_v2
