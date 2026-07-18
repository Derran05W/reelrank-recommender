#pragma once

#include <cstdint>

#include "rr/domain/ids.hpp"

namespace rr {

// Hidden per-reel archetype state (V2 TDD 4.4 + 5, Phase 13), owned SOLELY by the simulator
// (D18): the archetype a reel was sampled from and the archetype-conditioned parameters later
// phases' behaviour model consumes (P14 latent reactions, P16 session dynamics, P20 retention).
// No archetype label or parameter ever reaches recommender-visible state — the ranker sees only
// the resulting Reel attribute values; the include-graph guard enforces the isolation
// structurally. Index-aligned with GeneratedDataset::reels; populated by augmentReelsV2 only
// when realism.content_v2 is on (empty vector under gate-off, D17).
struct HiddenReelState {
    ReelId reelId{};

    // Index into RealismConfig::archetypes (the config-driven catalog, D24).
    uint32_t archetypeIndex = 0;

    // Additive bias applied to the latent immediateSatisfaction / regret computed in Phase 14.
    // Ragebait: negative satisfaction bias, positive regret bias; useful/satisfying: positive
    // satisfaction bias.
    float satisfactionBias = 0.0f;
    float regretBias = 0.0f;

    // Clickbait retention shape (V2 TDD 4.4): openingHook is the strength of the early-watch
    // boost, retentionDecay the post-hook abandonment rate. Zero for non-clickbait archetypes.
    float openingHook = 0.0f;
    float retentionDecay = 0.0f;

    // Niche-treasure cohort (V2 TDD 4.4): the reel is highly satisfying only to users whose
    // deterministic hash01(userId) falls within +-nicheCohortWidth of nicheCohortCentre (the
    // Phase 10 cohort-hash mechanism). Width 0 = not a niche reel.
    float nicheCohortCentre = 0.0f;
    float nicheCohortWidth = 0.0f;

    // Comfort-content return-probability bonus (V2 TDD 4.4), consumed by P20 retention.
    float comfortReturnBonus = 0.0f;
};

} // namespace rr
