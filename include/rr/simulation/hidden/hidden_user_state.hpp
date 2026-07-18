#pragma once

#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"

namespace rr {

// Ground-truth user state owned SOLELY by the simulator (design decisions D11/D18). The
// recommender must never see this type — the "recommender never accesses hidden preference"
// property is thus a structural, compile-time guarantee, enforced by the include-graph guard
// (scripts/check_hidden_isolation.py): nothing under recommendation/, candidate_sources/, or
// learning/ may reach a simulation/hidden/ header. Behavioural traits (TDD 9.3) are sampled per
// user in Phase 2 and consumed by the behaviour model in Phase 3; Realism V2 channels and traits
// (V2 TDD 4.2 + 5) are sampled by augmentUsersV2 in Phase 13.
struct HiddenUserState {
    UserId userId;

    // True latent preference vector (L2-normalized). The simulator scores reels against this; the
    // recommender never sees it.
    Embedding hiddenPreference;

    // Ground-truth topics (2-5, distinct) whose weighted mix formed hiddenPreference. Recorded so
    // the simulator/tests can reason about the user's true interests. Not visible to the
    // recommender.
    std::vector<TopicId> preferredTopics{};

    // --- Per-user behavioural traits (TDD 9.3 user-variation axes). Each field documents what it
    //     modulates and its valid sampling range; generateUsers samples within these ranges and
    //     the tests verify every generated value falls inside them. ---

    // Preference concentration: peakedness of the topic weights that formed hiddenPreference.
    // Higher => one topic dominates; lower => a flatter mix. Valid range [0.5, 4.0].
    float preferenceConcentration = 1.0f;

    // Willingness to explore unfamiliar / off-preference content (probability-like). [0.0, 1.0].
    float exploreWillingness = 0.0f;

    // Average session length: mean number of reels the user watches per session. [5.0, 40.0].
    float avgSessionLength = 0.0f;

    // Baseline propensity to like a reel, before affinity/quality modulation. [0.02, 0.25].
    float likePropensity = 0.0f;

    // Baseline propensity to share a reel, before affinity/quality modulation. [0.0, 0.10].
    float sharePropensity = 0.0f;

    // Tolerance for long videos: higher => more willing to keep watching long durations.
    // [0.0, 1.0].
    float durationTolerance = 0.0f;

    // Preference stability: how slowly the hidden preference drifts over time (higher => more
    // stable). Consumed by preference-drift logic in later phases. [0.0, 1.0].
    float preferenceStability = 0.0f;

    // --- Realism V2 hidden preference channels and traits (V2 TDD 4.2 + 5, Phase 13) ----------
    // Sampled by augmentUsersV2 from stream "users-v2" only when realism.content_v2 is on; the
    // defaults below are the gate-off values (zero V2 draws occur under gate-off, D17). Sampling
    // ranges are documented as the single source of truth in rr::userTraitsV2
    // (user_augmenter_v2.hpp); tests verify every generated value falls inside them.

    // Per-modality preference embeddings (topic channel = hiddenPreference above). L2-normalized
    // mixes of the shared ModalitySpaces centres, dimension = simulation.dimensions.
    Embedding visualPreference{};
    Embedding musicPreference{};
    Embedding emotionalPreference{};

    // Scalar content-value preferences and susceptibility traits (V2 TDD 4.2).
    float usefulnessPreference = 0.0f;
    float humourPreference = 0.0f;
    float controversyTolerance = 0.0f;
    float noveltySeeking = 0.0f;
    float clickbaitSusceptibility = 0.0f;
    float informationTolerance = 0.0f;

    // Language affinity: the user's primary language (drawn from the same skewed global
    // distribution as reel languages) and how tolerant they are of mismatched-language content.
    LanguageId primaryLanguage{};
    float languageMismatchTolerance = 0.0f;

    // --- Forward traits: generated in Phase 13, consumed by the named later phase (V2 TDD 5) ---
    float repetitionTolerance = 0.0f;  // P16-17 fatigue heterogeneity
    float noveltyTolerance = 0.0f;     // P16-17 fatigue heterogeneity
    float creatorLoyalty = 0.0f;       // P16-17 creator-fatigue immunity
    float habitStrength = 0.0f;        // P20 retention
    float platformTrust = 0.0f;        // P20 retention
    float baselineDailyUsage = 0.0f;   // P20 retention (expected sessions per simulated day)
    float preferencePlasticity = 0.0f; // P20 exposure-driven preference evolution
};

} // namespace rr
