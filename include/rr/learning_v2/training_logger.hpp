#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <utility>
#include <vector>

#include "rr/domain/behaviour_outcome.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/learning_v2/training_log_schema.hpp"
#include "rr/learning_v2/training_matrix.hpp"
#include "rr/recommendation/orchestrator.hpp"

namespace rr {

// ================================================================================================
// Phase 22 training-data LOGGER (contracts docs/design/P22-CONTRACTS.md §2/§3). Writes the
// leak-proof training log under <run-dir>/training_log/ (requests.csv, candidates-partNNNN.csv,
// outcomes-partNNNN.csv, schema.json) for the event-driven world only.
//
// D18 PURITY BY CONSTRUCTION (contracts §3): this is a NON-carve-out TU under the D18-scanned
// learning_v2 roots (scripts/check_hidden_isolation.py). It sees ONLY recommender-visible /
// observable inputs — the request, the observable User, the served-time RankingCapture (pool +
// FeatureVectors + scores + provenance), the RankedReels, and the observable InteractionEvent /
// BehaviourOutcome. It NEVER includes simulation/hidden/, so no latent/archetype/trust/fatigue
// value can reach a logged column. The one hidden-derived signal (the noisy survey) is emitted by
// the SEPARATE evaluation-side SurveyWriter (carve-out), not here.
//
// SAMPLING DRAWS NO RNG (contracts §1): a request is logged iff the pinned, rng-free predicate
// learning_v2::logSampleSelected(requestId, salt, rate) holds — kLogSampleSalt with log_sample_rate
// for its SHOWN impressions (features + outcomes join) and kLogPoolSampleSalt with
// log_pool_sample_rate for the FULL ranked pool. The two salts are independent, so the decisions
// are independent, and neither perturbs a simulation stream (golden-tripwired like cohortHash01,
// D19). Both hooks recompute the predicate from the request id, so the logger is stateless across
// requests.
//
// STREAMING + ROTATION: files are streamed under <run-dir>/training_log/ during the run, opened
// lazily on the first row (header written then). candidates/outcomes rotate into contiguous
// part files (candidates-part0000.csv, -part0001.csv, ...) at log_max_rows_per_file DATA rows each;
// requests.csv and schema.json are single files. finish() flushes, ensures every table exists (an
// empty run still emits header-only files so package B's reader always finds them), and writes
// schema.json.
// ================================================================================================
class TrainingLogger {
  public:
    // `runDir` is the run's output directory (<algorithm>-seedN-STAMP); the log is written to
    // `<runDir>/training_log/`. Held for the whole run so streaming appends and finish() share it.
    TrainingLogger(const LearningV2Config &config, std::filesystem::path runDir);

    // P22-HOOK(ranking) sink (contracts §3, "after ranking: features + pool + shown capture").
    // Called once per SAMPLED feed request from the event runner. `requestId` is the request's id
    // (the event runner's per-request counter, joined by InteractionEvent::requestId at the outcome
    // hook). `effectiveEpsilon` is the exploration probability actually in effect for this request
    // (0 when exploration is off or before its enable-at-day gate). `capture` holds the SERVED-TIME
    // pool + per-candidate FeatureVectors + scores + provenance; `rankedFeed` is the shown feed
    // with positions. Emits the requests.csv row (both-rate union) and the matching candidates rows
    // — pool rows (pool_rank + feed position or -1 + shown flag) for a pool-sampled request, else
    // just the shown rows (with feed position) for a shown-only-sampled request.
    void onRequestRanked(std::uint64_t requestId, const RecommendationRequest &request,
                         const User &user, double effectiveEpsilon, const RankingCapture &capture,
                         const std::vector<RankedReel> &rankedFeed);

    // P22-HOOK(outcome) sink (contracts §3, "after each impression's InteractionEvent: outcome
    // append"). Appends the observable label row to outcomes.csv for a SHOWN-sampled impression,
    // joined on (request_id, reel_id). The completed/liked/shared/followed/not_interested labels
    // come from `outcome` (InteractionEvent::type is lossy for simultaneous signals); the watch /
    // V2 engagement / exit observables come from `event`. Called for every impression; the logger
    // filters to shown-sampled requests via the pinned predicate (no-op for the rest).
    void onImpressionOutcome(const InteractionEvent &event, const BehaviourOutcome &outcome);

    // P22-HOOK(finish) sink (contracts §3, "run end: flush + schema.json"). Flushes/closes the CSV
    // streams, ensures every §2 table exists (header-only if the run logged none), and writes
    // schema.json (schema_version + column allowlists + salts + config echo + file layout).
    void finish();

    // --- Phase 23 in-memory training matrix (contracts §3) -------------------------------------
    // Kept ONLY when learning_v2.learned_ranker is on (a P22-only training-log run allocates
    // nothing new). The three sinks above ALSO stream the SAME sampled shown rows into `matrix_`
    // when keepMatrix_: onRequestRanked stores the served features, onImpressionOutcome joins the
    // observable labels, and onSurvey (below) joins the sanctioned survey likert. The map is keyed
    // (request_id, reel_id) and ORDERED, so snapshotMatrix() iterates it in canonical order with no
    // hash-map-order dependence — training is bit-reproducible across platforms (D8/§5). Memory: at
    // medium scale the map holds the run's sampled shown candidates (~100 bytes payload/row +
    // std::map node overhead) — order ~10–30 MB, documented; float storage keeps it compact. The
    // Retrainer consumes snapshotMatrix()'s complete rows; the runner gates retrains on
    // matrixCompleteRows() >= min_training_rows.

    // P23 survey sink: joins the OBSERVABLE survey likert (1..5) to the matrix row for a surveyed
    // shown impression (contracts §2 — the likert is observable explicit feedback, not hidden
    // state; the hidden immediateSatisfaction never reaches here, D18). No-op unless keepMatrix_
    // and the (request_id, reel_id) row already exists (it does: features were captured at
    // ranking). The runner calls this only when the survey fired for a sampled shown impression.
    void onSurvey(std::uint64_t requestId, ReelId reelId, int likert);

    // Number of TRAINABLE rows (hasFeatures && hasOutcome) accumulated so far — the runner's
    // retrain-eligibility check. O(1) (a running counter, not a scan).
    std::size_t matrixCompleteRows() const { return matrixCompleteRows_; }

    // Snapshot the complete rows in canonical (request_id, reel_id) order for a retrain. O(matrix);
    // called only at retrain boundaries (a handful of times per run).
    std::vector<learning_v2::ShownFeatureRow> snapshotMatrix() const;

  private:
    void ensureDir();
    void openRequests();
    void openCandidatesPart(std::uint32_t index);
    void openOutcomesPart(std::uint32_t index);

    LearningV2Config config_;
    std::filesystem::path trainingLogDir_; // <runDir>/training_log

    // Lazily-opened streams + per-part row counters for rotation (contracts §2).
    std::ofstream requestsOut_;
    std::ofstream candidatesOut_;
    std::ofstream outcomesOut_;
    std::uint64_t candidatesRowsInPart_ = 0;
    std::uint64_t outcomesRowsInPart_ = 0;
    std::uint32_t candidatesPartIndex_ = 0;
    std::uint32_t outcomesPartIndex_ = 0;
    bool requestsOpened_ = false;
    bool candidatesOpened_ = false;
    bool outcomesOpened_ = false;

    // Phase 23 in-memory matrix (see the public section). keepMatrix_ == config.learnedRanker.
    bool keepMatrix_ = false;
    std::map<std::pair<std::uint64_t, std::uint64_t>, learning_v2::ShownFeatureRow> matrix_;
    std::size_t matrixCompleteRows_ = 0; // running count of hasFeatures && hasOutcome rows
};

} // namespace rr
