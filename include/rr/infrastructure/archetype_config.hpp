#pragma once

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace rr {

// One entry of the config-driven archetype catalog (V2 TDD 4.4, D24): a named set of
// attribute-distribution parameters plus a mixture weight. The catalog is DATA, not code —
// adding an archetype never requires recompiling — and the archetype identity itself is hidden
// (D18): augmentReelsV2 samples an archetype per reel from stream "archetypes", shapes the
// reel's serving-visible attribute draws with these parameters, and records the index plus the
// hidden parameters in HiddenReelState. The ranker sees only the resulting attribute values.
//
// PACKAGE-A OWNERSHIP, FROZEN CONTRACT: package A extends this struct with the distribution
// parameters that shape each attribute draw (means/spreads for the [0,1] scalars, modality
// coherence, hidden satisfaction/regret/hook/niche/comfort parameters), all with defaults and
// covered by the defaulted operator==. The following must NOT change, because
// config.{hpp,cpp} (Fable-owned) depends on them: the struct name, the `name` and `weight`
// members, default-constructibility, operator==, and the three declarations below.
//
// All parameters are `double` (the JSON numeric type, so config values round-trip exactly through
// to_json/from_json) and cast to float only when a draw is materialized. The defaults below are a
// deliberately NEUTRAL archetype: a config that names an archetype but omits parameter keys (or a
// default-constructed ArchetypeSpec) gets middle-of-the-road, low-controversy content. The eight
// shipped archetypes (defaultArchetypeCatalog) override these with DISTINCT signatures.
//
// How augmentReelsV2 (reel_augmenter_v2.cpp) consumes each field is documented there, at the
// pinned per-reel draw sequence; in brief: `*Mean` fields are the mean of a clamped-gaussian draw
// with stddev `scalarSpread`; `productionQualityShift` is added to the reel's V1 intrinsicQuality
// (correlated, not aliased); `*Coherence` fields in [0, 1] set how tightly each modality embedding
// tracks its sampled style centre (1 = on the centre, 0 = pure noise); the six hidden params are
// copied verbatim into HiddenReelState for later phases (P14/P16/P20), with nicheCohortCentre
// drawn per reel only when nicheCohortWidth > 0.
struct ArchetypeSpec {
    std::string name;
    // Relative mixture weight (> 0; normalized over the catalog at sampling time).
    double weight = 1.0;

    // --- Serving-visible [0, 1] scalar attribute distributions (V2 TDD 4.1) ------------------
    // Mean of each attribute's clamped-gaussian draw; `scalarSpread` is the shared stddev.
    double usefulnessMean = 0.45;
    double humourMean = 0.45;
    double noveltyMean = 0.45;
    double controversyMean = 0.20;
    double clickbaitStrengthMean = 0.10;
    double informationDensityMean = 0.40;
    double emotionalIntensityMean = 0.40;
    // productionQuality is drawn as clamp01(reel.intrinsicQuality + productionQualityShift +
    // scalarSpread * gaussian()): correlated with the V1 quality the behaviour model still reads,
    // shifted per archetype (e.g. polished-irrelevant lifts polish above substance). No *Mean of
    // its own — it rides on intrinsicQuality by design (reel.hpp reconciliation note).
    double productionQualityShift = 0.0;
    // Shared stddev of the eight scalar draws (>= 0). One knob, not eight (D24
    // no-premature-config).
    double scalarSpread = 0.15;

    // --- Modality-embedding coherence, per channel, in [0, 1] (V2 TDD 4.1/4.2) ---------------
    // Blend weight of the sampled style centre against unit-scaled gaussian noise; higher = the
    // modality embedding sits closer to a single centre. background_music sets a high
    // musicCoherence so music embeddings cluster tightly regardless of topic.
    double visualCoherence = 0.55;
    double musicCoherence = 0.55;
    double emotionalCoherence = 0.55;

    // --- Hidden archetype parameters, copied verbatim into HiddenReelState (D18) -------------
    // Simulator-only; never reach recommender-visible state. Consumed by later phases.
    double satisfactionBias = 0.0; // additive bias on latent immediate satisfaction (P14)
    double regretBias = 0.0;       // additive bias on latent regret (P14)
    double openingHook = 0.0;      // clickbait early-watch boost strength (P14/P16)
    double retentionDecay = 0.0;   // clickbait post-hook abandonment rate (P14/P16)
    double nicheCohortWidth = 0.0; // niche-treasure cohort half-width in hash01 space (>0 => niche)
    double comfortReturnBonus = 0.0; // comfort-content return-probability bonus (P20)

    bool operator==(const ArchetypeSpec &) const = default;
};

// D6 serialization: from_json rejects unknown keys (catches typos), validates weight > 0, and
// requires `name` to be present and non-empty; missing parameter keys keep the defaults.
void to_json(nlohmann::json &j, const ArchetypeSpec &a);
void from_json(const nlohmann::json &j, ArchetypeSpec &a);

// The eight V2 TDD 4.4 archetypes, transliterated to distribution parameters: genuinely
// satisfying, useful, ragebait, clickbait, comfort, polished-irrelevant, niche-treasure,
// background-music. This is the default value of RealismConfig::archetypes.
std::vector<ArchetypeSpec> defaultArchetypeCatalog();

} // namespace rr
