#pragma once

#include <cstdint>
#include <vector>

#include "rr/learning_v2/learned_ranker.hpp"
#include "rr/learning_v2/training_matrix.hpp"

// ================================================================================================
// Phase 23 in-loop RETRAINER (contracts docs/design/P23-CONTRACTS.md §3). Turns the
// TrainingLogger's in-memory shown-row matrix into a fresh LearnedModels bundle using the P22
// in-house learners (LogisticRegression / LinearRegression), at retrain_epochs, with a
// master-seed-derived, per-version salted seed so the same run seed reproduces a bit-identical
// model sequence (test-enforced §5).
//
// DETERMINISM (contracts §3): each retrain version v derives one seed = splitmix64(runSeed ^
// (GAMMA·v)); every target's learner then forks its "model-init"/"training-split" streams (the P22
// internals) from that seed, so the six models of version v are jointly reproducible. Wall cost is
// measured by the RUNNER around retrain() with steady_clock (D9 — outside simulated time); it is
// not part of the determinism guarantee (like latency_metrics.csv). Rows are consumed in the
// caller's canonical (request_id, reel_id) order (TrainingLogger::snapshotMatrix sorts them), so
// training is hash-map-order-independent.
// ================================================================================================

namespace rr::learning_v2 {

// Derive the per-version learner master seed (contracts §3). Pure, rng-free (splitmix64 mixer): a
// deterministic function of the run seed and the version, never a simulation-stream draw, so it
// cannot perturb any world stream (D8/D19). Exposed for the determinism unit test.
std::uint64_t retrainVersionSeed(std::uint64_t runSeed, int version);

class Retrainer {
  public:
    Retrainer(std::uint64_t runSeed, std::uint32_t epochs);

    // Train the six §4.21 targets on `rows` (already filtered to complete rows and sorted into the
    // canonical order by the caller). A target with too few positives/examples (P22 honest-SKIP,
    // kMinPositivesToTrain) is left absent — its value term then contributes 0. `version` (>=1)
    // sets the per-version seed and stamps the bundle. `ready` is true iff at least one model
    // trained.
    LearnedModels retrain(const std::vector<ShownFeatureRow> &rows, int version) const;

  private:
    std::uint64_t runSeed_;
    std::uint32_t epochs_;
};

} // namespace rr::learning_v2
