#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>

#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"

namespace rr {

// ================================================================================================
// Phase 22 explicit-feedback SURVEY writer (contracts docs/design/P22-CONTRACTS.md §2/§3, V2 TDD
// 4.19). Emits survey.csv (user_id, reel_id, request_id, timestamp, likert) under
// <run-dir>/training_log/ — the ONLY hidden-derived training table, clearly labeled in schema.json
// and exempted (documented columns only) from the purity audit's forbidden-substring scan.
//
// D18 CARVE-OUT: unlike the TrainingLogger, this writer lives in evaluation/ precisely because it
// READS the latent immediateSatisfaction (oracle-flavoured, the sanctioned evaluation carve-out,
// D18). Keep it OUT of the learning_v2 roots so the include-graph guard is not tripped.
//
// TWO-DRAW "explicit-feedback" CONTRACT (contracts §1, D19): when survey.enabled, EVERY shown
// impression makes EXACTLY TWO draws on the D19-pinned "explicit-feedback" rng stream (owned by the
// runner, passed in) —
//   (1) bernoulli(survey.sample_rate)      -> is this impression surveyed?
//   (2) gaussian() * survey.noise_sd       -> additive noise, then likert = quantize(clamp(
//                                             immediateSatisfaction + noise) to 1..5)  (mapping
//                                             documented at maybeSurvey's implementation)
// Both draws happen UNCONDITIONALLY per shown impression (before any early return) so the draw
// COUNT is impression-aligned and stream-stable; only draw (1)'s OUTCOME decides whether a row is
// written. When disabled the writer is never constructed and the stream is NEVER touched (zero
// draws), so V1 behaviour streams stay byte-identical.
// ================================================================================================
class SurveyWriter {
  public:
    // `runDir` is the run's output directory; survey.csv is written to `<runDir>/training_log/`.
    SurveyWriter(const LearningV2Config &config, std::filesystem::path runDir);

    // P22-HOOK(outcome) sink: called once per SHOWN impression (see the two-draw contract above).
    // Draws twice on `explicitFeedback`, and — iff draw (1) fires — appends one survey.csv row.
    void maybeSurvey(UserId userId, ReelId reelId, std::uint64_t requestId, Timestamp timestamp,
                     float immediateSatisfaction, Rng &explicitFeedback);

    // P22-HOOK(finish) sink: flush survey.csv at run end. Ensures the file exists (header-only if
    // no impression was surveyed) so the schema.json-declared table is always present.
    void finish();

  private:
    void ensureOpen();

    LearningV2Config config_;
    std::filesystem::path trainingLogDir_; // <runDir>/training_log
    std::ofstream surveyOut_;
    bool opened_ = false;
};

} // namespace rr
