#include "rr/simulation/reel_augmenter_v2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "rr/core/embedding.hpp"
#include "rr/simulation/language.hpp"

namespace rr {
namespace {

// --- Modality style-centre counts (D24 no-premature-config named constants) --------------------
// K L2-normalized random unit centres per modality, shared by every reel and (package B) by the
// hidden user modality preferences, so the two sides have learnable common structure. Music gets
// the most centres (the richest distinct-style space: genre/tempo/mood combine into many clusters),
// visual fewer, emotional tone the fewest (a handful of broad tones). No planned experiment varies
// these, so they stay constants here rather than on the config surface; bump them if a future phase
// needs to sweep modality granularity. Widening simulation.reels never changes the centres because
// generateModalitySpaces draws them FIRST on "reels-v2", before any per-reel draw (pinned order).
constexpr uint32_t kVisualCentres = 12;
constexpr uint32_t kMusicCentres = 16;
constexpr uint32_t kEmotionalCentres = 8;

float clamp01(double v) { return static_cast<float>(std::clamp(v, 0.0, 1.0)); }

// A random L2-normalized unit vector: `dim` gaussian draws then normalize (the standard way to
// sample uniformly on the unit sphere). Consumes exactly `dim` draws from `rng`.
Embedding randomUnitVector(uint32_t dim, Rng &rng) {
    Embedding e(dim, 0.0f);
    for (uint32_t d = 0; d < dim; ++d) {
        e[d] = static_cast<float>(rng.gaussian());
    }
    normalize(e);
    return e;
}

std::vector<Embedding> generateCentres(uint32_t count, uint32_t dim, Rng &rng) {
    std::vector<Embedding> centres;
    centres.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        centres.push_back(randomUnitVector(dim, rng));
    }
    return centres;
}

// One reel modality embedding: a coherence-weighted blend of a uniformly sampled style centre with
// unit-scaled gaussian noise, then L2-normalized. `coherence` in [0, 1] is the centre weight (1 =
// on the centre, 0 = pure noise). The noise per-component stddev is 1/sqrt(dim), so the noise term
// has ~unit expected norm, comparable to the unit centre, making the cosine-to-centre a smooth
// increasing function of coherence. FIXED per-reel draw count on `rng`, INDEPENDENT of the
// archetype: exactly 1 uniformInt (centre pick) + `dim` gaussians (noise). `coherence` changes the
// values, never the draw count — the determinism/stream-alignment contract (see augmentReelsV2).
Embedding modalityEmbedding(const std::vector<Embedding> &centres, double coherence, uint32_t dim,
                            Rng &rng) {
    // centres is always non-empty (counts above are > 0); uniformInt is called unconditionally so
    // the draw count stays fixed.
    const std::size_t centreIdx = static_cast<std::size_t>(rng.uniformInt(centres.size()));
    const Embedding &centre = centres[centreIdx];
    const double noisePerDim = 1.0 / std::sqrt(static_cast<double>(dim));

    Embedding e(dim, 0.0f);
    for (uint32_t d = 0; d < dim; ++d) {
        const double noise = rng.gaussian();
        e[d] = static_cast<float>(coherence * static_cast<double>(centre[d]) +
                                  (1.0 - coherence) * noisePerDim * noise);
    }
    normalize(e);
    return e;
}

// Prefix-sum a positive weight vector; returns the total. Used for both archetype mixture sampling
// and language sampling (draw u in [0, total), return the first index whose prefix sum exceeds u).
std::vector<double> cumulative(const std::vector<double> &weights, double &total) {
    std::vector<double> cum(weights.size());
    double running = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        running += weights[i];
        cum[i] = running;
    }
    total = running;
    return cum;
}

// First index whose cumulative weight strictly exceeds `u` (u in [0, cum.back())). Clamped to the
// last index so a floating-point u == cum.back() can never fall off the end.
std::uint32_t sampleCumulative(const std::vector<double> &cum, double u) {
    std::uint32_t idx = 0;
    while (idx + 1 < cum.size() && u >= cum[idx]) {
        ++idx;
    }
    return idx;
}

} // namespace

ModalitySpaces generateModalitySpaces(const SimulationConfig &config, const RealismConfig &realism,
                                      Rng &reelsV2Rng) {
    (void)realism; // centre counts are content structure (named constants), not config-driven.
    const uint32_t dim = config.dimensions;

    // Pinned draw order: visual centres, then music, then emotional — the FIRST draws on
    // "reels-v2", before augmentReelsV2's per-reel draws. This ordering plus the fixed counts is
    // why widening simulation.reels leaves the centres (and thus package B's user preferences,
    // which blend the SAME centres) byte-identical.
    ModalitySpaces spaces;
    spaces.visualCentres = generateCentres(kVisualCentres, dim, reelsV2Rng);
    spaces.musicCentres = generateCentres(kMusicCentres, dim, reelsV2Rng);
    spaces.emotionalCentres = generateCentres(kEmotionalCentres, dim, reelsV2Rng);
    return spaces;
}

void augmentReelsV2(std::vector<Reel> &reels, std::vector<HiddenReelState> &hiddenReelStates,
                    const ModalitySpaces &spaces, const SimulationConfig &config,
                    const RealismConfig &realism, Rng &archetypesRng, Rng &reelsV2Rng) {
    const uint32_t dim = config.dimensions;

    // Precompute the archetype mixture and language cumulative distributions once (pure functions
    // of the config, no rng).
    const std::vector<ArchetypeSpec> &catalog = realism.archetypes;
    std::vector<double> archWeights(catalog.size());
    for (std::size_t a = 0; a < catalog.size(); ++a) {
        archWeights[a] = catalog[a].weight;
    }
    double archTotal = 0.0;
    const std::vector<double> archCum = cumulative(archWeights, archTotal);

    double langTotal = 0.0;
    const std::vector<double> langCum = cumulative(languageWeights(realism.languages), langTotal);

    hiddenReelStates.clear();
    hiddenReelStates.reserve(reels.size());

    // PINNED PER-REEL DRAW CONTRACT (independent of which archetype is sampled — conditional
    // PARAMETERS, never conditional draw COUNTS, so editing the catalog changes values but never
    // stream alignment; two catalogs that differ only in non-weight parameters therefore produce
    // the same archetype sequence and byte-identical language ids):
    //   "archetypes" stream, per reel: exactly 1 uniform01 (weighted archetype pick).
    //   "reels-v2"  stream, per reel, in THIS order:
    //     1. visual modality embedding   : 1 uniformInt + dim gaussians
    //     2. music modality embedding     : 1 uniformInt + dim gaussians
    //     3. emotional modality embedding : 1 uniformInt + dim gaussians
    //     4. eight scalars, in field order (usefulness, humour, novelty, productionQuality,
    //        controversy, clickbaitStrength, informationDensity, emotionalIntensity): 8 gaussians
    //     5. language                     : 1 uniform01
    //     6. niche cohort centre          : 1 uniform01 (ALWAYS drawn; stored only when width > 0)
    //   => 3 uniformInt + (3*dim + 8) gaussians + 2 uniform01 on "reels-v2" per reel, fixed.
    for (Reel &r : reels) {
        // (archetypes stream) weighted archetype selection.
        const std::uint32_t archIdx =
            sampleCumulative(archCum, archetypesRng.uniform01() * archTotal);
        const ArchetypeSpec &A = catalog[archIdx];

        // (reels-v2 stream) 1-3: modality embeddings (centre blend + noise, L2-normalized).
        r.visualStyleEmbedding =
            modalityEmbedding(spaces.visualCentres, A.visualCoherence, dim, reelsV2Rng);
        r.musicEmbedding =
            modalityEmbedding(spaces.musicCentres, A.musicCoherence, dim, reelsV2Rng);
        r.emotionalToneEmbedding =
            modalityEmbedding(spaces.emotionalCentres, A.emotionalCoherence, dim, reelsV2Rng);

        // (reels-v2 stream) 4: eight [0,1] scalars, archetype mean + shared-spread gaussian noise.
        // productionQuality is the ONLY one tied to a V1 field: it is drawn CORRELATED with the
        // reel's existing intrinsicQuality (read-only) plus the archetype's polish shift, so V1
        // consumers keep reading intrinsicQuality unchanged while V2 exposes productionQuality
        // (reel.hpp reconciliation). Every draw is one gaussian regardless of archetype.
        r.usefulness = clamp01(A.usefulnessMean + A.scalarSpread * reelsV2Rng.gaussian());
        r.humour = clamp01(A.humourMean + A.scalarSpread * reelsV2Rng.gaussian());
        r.novelty = clamp01(A.noveltyMean + A.scalarSpread * reelsV2Rng.gaussian());
        r.productionQuality =
            clamp01(static_cast<double>(r.intrinsicQuality) + A.productionQualityShift +
                    A.scalarSpread * reelsV2Rng.gaussian());
        r.controversy = clamp01(A.controversyMean + A.scalarSpread * reelsV2Rng.gaussian());
        r.clickbaitStrength =
            clamp01(A.clickbaitStrengthMean + A.scalarSpread * reelsV2Rng.gaussian());
        r.informationDensity =
            clamp01(A.informationDensityMean + A.scalarSpread * reelsV2Rng.gaussian());
        r.emotionalIntensity =
            clamp01(A.emotionalIntensityMean + A.scalarSpread * reelsV2Rng.gaussian());

        // (reels-v2 stream) 5: language, one uniform draw against the shared skewed distribution.
        r.language = LanguageId{sampleCumulative(langCum, reelsV2Rng.uniform01() * langTotal)};

        // (reels-v2 stream) 6: niche cohort centre in the Phase 10 hash01 space [0, 1). Drawn
        // UNCONDITIONALLY to keep the per-reel draw count fixed; only recorded when the archetype
        // is actually a niche archetype (width > 0), else left 0 (not a niche reel).
        const double nicheCentre = reelsV2Rng.uniform01();

        HiddenReelState h;
        h.reelId = r.id;
        h.archetypeIndex = archIdx;
        h.satisfactionBias = static_cast<float>(A.satisfactionBias);
        h.regretBias = static_cast<float>(A.regretBias);
        h.openingHook = static_cast<float>(A.openingHook);
        h.retentionDecay = static_cast<float>(A.retentionDecay);
        h.nicheCohortWidth = static_cast<float>(A.nicheCohortWidth);
        h.nicheCohortCentre = (A.nicheCohortWidth > 0.0) ? static_cast<float>(nicheCentre) : 0.0f;
        h.comfortReturnBonus = static_cast<float>(A.comfortReturnBonus);
        hiddenReelStates.push_back(h);
    }
}

} // namespace rr
