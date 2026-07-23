#pragma once

#include <array>
#include <cstdint>

#include "rr/learning_v2/sgd_common.hpp"

// ================================================================================================
// Phase 23 in-memory training-matrix row (contracts §3). Split into this tiny header so the
// TrainingLogger (which accumulates the matrix) does not have to pull in the learners / the
// LearnedRanker — only the Retrainer, which turns rows into models, includes both this and
// learned_ranker.hpp. Depends on nothing but <array>/<cstdint> and sgd_common (kNumFeatures).
// ================================================================================================

namespace rr::learning_v2 {

// One joined shown training example held in the TrainingLogger's in-memory matrix (kept only when
// learning_v2.learned_ranker is on). Features stored as float (the FeatureVector precision) to keep
// the matrix compact — ~100 bytes of payload per row. Only the SIX §4.21 targets' labels are kept
// (completed/liked are P22 offline-only, never served). `likert` is 0 when the row was not surveyed
// (satisfaction then skips it). Assembled across three hooks: features at ranking, the binary/watch
// labels at the impression outcome, the survey likert at the survey draw. A row is TRAINABLE once
// hasFeatures && hasOutcome (a shown candidate the user actually consumed produced an outcome).
struct ShownFeatureRow {
    std::array<float, kNumFeatures> features{};
    float watchRatio = 0.0F;        // watch_ratio target (from the InteractionEvent)
    std::uint8_t shared = 0;        // shared target
    std::uint8_t followed = 0;      // followed target
    std::uint8_t notInterested = 0; // not_interested target (the pRegret proxy)
    std::uint8_t sessionExit = 0;   // session_exit = observed_exit_after_impression
    std::uint8_t likert = 0;        // survey satisfaction 1..5, 0 = not surveyed
    bool hasFeatures = false;       // ranking hook fired
    bool hasOutcome = false;        // impression-outcome hook fired (row is trainable)
};

} // namespace rr::learning_v2
