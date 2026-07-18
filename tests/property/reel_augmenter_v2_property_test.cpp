#include "rr/simulation/reel_augmenter_v2.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "rr/infrastructure/config.hpp"
#include "rr/simulation/dataset_generator.hpp"

// Phase 13 reel-side property tests (V2 TDD 4.1/4.4, 7): determinism, the fixed-per-reel-draw-count
// stream-alignment contract, reel-count extension stability, and V1-field non-perturbation — each
// swept across >= 20 seeds (design decision D7 property style).

namespace {

constexpr std::uint64_t kNumSeeds = 25; // >= 20 required

rr::SimulationConfig propSim() {
    rr::SimulationConfig c;
    c.reels = 200;
    c.users = 30;
    c.creators = 50;
    c.topics = 16;
    c.dimensions = 32;
    return c;
}

rr::RealismConfig realismOn() {
    rr::RealismConfig r;
    r.contentV2 = true;
    r.languages = 8;
    return r;
}

// A catalog that differs from the default ONLY in non-weight parameters (every mixture weight is
// preserved). Under the fixed-draw-count contract this must leave the archetype sequence and the
// per-reel language draws byte-identical while changing the attribute VALUES.
rr::RealismConfig realismPerturbedNonWeight() {
    rr::RealismConfig r = realismOn();
    for (rr::ArchetypeSpec &a : r.archetypes) {
        // a.weight deliberately untouched.
        a.usefulnessMean = 0.90;
        a.humourMean = 0.10;
        a.noveltyMean = 0.80;
        a.controversyMean = 0.05;
        a.clickbaitStrengthMean = 0.70;
        a.informationDensityMean = 0.15;
        a.emotionalIntensityMean = 0.60;
        a.productionQualityShift = 0.20;
        a.scalarSpread = 0.20;
        a.visualCoherence = 0.30;
        a.musicCoherence = 0.70;
        a.emotionalCoherence = 0.40;
        a.satisfactionBias = 0.50;
        a.regretBias = -0.50;
        a.openingHook = 0.30;
        a.retentionDecay = 0.30;
        a.nicheCohortWidth = 0.25;
        a.comfortReturnBonus = 0.20;
    }
    return r;
}

bool reelV2Equal(const rr::Reel &a, const rr::Reel &b) {
    return a.visualStyleEmbedding == b.visualStyleEmbedding &&
           a.musicEmbedding == b.musicEmbedding &&
           a.emotionalToneEmbedding == b.emotionalToneEmbedding && a.usefulness == b.usefulness &&
           a.humour == b.humour && a.novelty == b.novelty &&
           a.productionQuality == b.productionQuality && a.controversy == b.controversy &&
           a.clickbaitStrength == b.clickbaitStrength &&
           a.informationDensity == b.informationDensity &&
           a.emotionalIntensity == b.emotionalIntensity && a.language == b.language;
}

bool hiddenEqual(const rr::HiddenReelState &a, const rr::HiddenReelState &b) {
    return a.reelId == b.reelId && a.archetypeIndex == b.archetypeIndex &&
           a.satisfactionBias == b.satisfactionBias && a.regretBias == b.regretBias &&
           a.openingHook == b.openingHook && a.retentionDecay == b.retentionDecay &&
           a.nicheCohortCentre == b.nicheCohortCentre && a.nicheCohortWidth == b.nicheCohortWidth &&
           a.comfortReturnBonus == b.comfortReturnBonus;
}

} // namespace

// Same seed twice => byte-identical V2 attributes, hidden states, and modality centres.
TEST(ReelAugmenterV2PropertyTest, DeterministicAcrossSeeds) {
    const auto sim = propSim();
    const auto realism = realismOn();
    for (std::uint64_t seed = 0; seed < kNumSeeds; ++seed) {
        SCOPED_TRACE(testing::Message() << "seed=" << seed);
        const auto a = rr::generateDataset(sim, realism, seed);
        const auto b = rr::generateDataset(sim, realism, seed);
        ASSERT_EQ(a.reels.size(), b.reels.size());
        ASSERT_EQ(a.hiddenReelStates.size(), b.hiddenReelStates.size());
        EXPECT_EQ(a.modalitySpaces.visualCentres, b.modalitySpaces.visualCentres);
        EXPECT_EQ(a.modalitySpaces.musicCentres, b.modalitySpaces.musicCentres);
        EXPECT_EQ(a.modalitySpaces.emotionalCentres, b.modalitySpaces.emotionalCentres);
        for (std::size_t i = 0; i < a.reels.size(); ++i) {
            EXPECT_TRUE(reelV2Equal(a.reels[i], b.reels[i])) << "reel " << i;
            EXPECT_TRUE(hiddenEqual(a.hiddenReelStates[i], b.hiddenReelStates[i]))
                << "hidden " << i;
        }
    }
}

// Fixed per-reel draw count => catalogs differing only in non-weight parameters produce the SAME
// archetype sequence (weights unchanged) AND byte-identical language ids (stream stays aligned),
// while the attribute values genuinely change (the test is not vacuous).
TEST(ReelAugmenterV2PropertyTest, NonWeightCatalogEditsPreserveSequenceAndLanguage) {
    const auto sim = propSim();
    const auto base = realismOn();
    const auto perturbed = realismPerturbedNonWeight();
    for (std::uint64_t seed = 0; seed < kNumSeeds; ++seed) {
        SCOPED_TRACE(testing::Message() << "seed=" << seed);
        const auto a = rr::generateDataset(sim, base, seed);
        const auto b = rr::generateDataset(sim, perturbed, seed);
        ASSERT_EQ(a.reels.size(), b.reels.size());
        bool anyValueChanged = false;
        for (std::size_t i = 0; i < a.reels.size(); ++i) {
            // Archetype sequence identical (same mixture weights, same "archetypes" draws).
            EXPECT_EQ(a.hiddenReelStates[i].archetypeIndex, b.hiddenReelStates[i].archetypeIndex)
                << "archetype drifted at reel " << i;
            // Language ids identical (fixed draw count keeps "reels-v2" aligned across catalogs).
            EXPECT_EQ(a.reels[i].language.value, b.reels[i].language.value)
                << "language drifted at reel " << i;
            anyValueChanged = anyValueChanged || a.reels[i].usefulness != b.reels[i].usefulness;
        }
        EXPECT_TRUE(anyValueChanged)
            << "perturbed catalog produced identical values (vacuous test)";
    }
}

// Augmenting N reels then N+M reels (same seed): the first N reels' V2 attributes and hidden states
// are identical, and the modality centres are identical — because the centres are drawn first (a
// fixed count) and each reel consumes a fixed number of draws thereafter.
TEST(ReelAugmenterV2PropertyTest, ReelCountExtensionKeepsFirstNIdentical) {
    rr::SimulationConfig small = propSim();
    rr::SimulationConfig big = propSim();
    big.reels = small.reels + 137; // extend by M
    const auto realism = realismOn();
    for (std::uint64_t seed = 0; seed < kNumSeeds; ++seed) {
        SCOPED_TRACE(testing::Message() << "seed=" << seed);
        const auto a = rr::generateDataset(small, realism, seed);
        const auto b = rr::generateDataset(big, realism, seed);
        ASSERT_EQ(a.reels.size(), small.reels);
        ASSERT_EQ(b.reels.size(), big.reels);
        // Centres are content structure, stable under reel-count changes (pinned first-draw order).
        EXPECT_EQ(a.modalitySpaces.visualCentres, b.modalitySpaces.visualCentres);
        EXPECT_EQ(a.modalitySpaces.musicCentres, b.modalitySpaces.musicCentres);
        EXPECT_EQ(a.modalitySpaces.emotionalCentres, b.modalitySpaces.emotionalCentres);
        for (std::size_t i = 0; i < a.reels.size(); ++i) {
            EXPECT_TRUE(reelV2Equal(a.reels[i], b.reels[i])) << "reel " << i;
            EXPECT_TRUE(hiddenEqual(a.hiddenReelStates[i], b.hiddenReelStates[i]))
                << "hidden " << i;
        }
    }
}

// Regenerating the V2 fields never changes the V1 fields (Phase 2 precedent + D17): the gate-on
// run's V1 reel fields are byte-identical to the gate-off run's across every seed.
TEST(ReelAugmenterV2PropertyTest, GateOnNeverPerturbsV1ReelFields) {
    const auto sim = propSim();
    rr::RealismConfig off;
    const auto on = realismOn();
    for (std::uint64_t seed = 0; seed < kNumSeeds; ++seed) {
        SCOPED_TRACE(testing::Message() << "seed=" << seed);
        const auto dsOff = rr::generateDataset(sim, off, seed);
        const auto dsOn = rr::generateDataset(sim, on, seed);
        ASSERT_EQ(dsOff.reels.size(), dsOn.reels.size());
        EXPECT_TRUE(dsOff.hiddenReelStates.empty()); // gate-off: zero V2 draws (D17)
        for (std::size_t i = 0; i < dsOff.reels.size(); ++i) {
            const rr::Reel &x = dsOff.reels[i];
            const rr::Reel &y = dsOn.reels[i];
            EXPECT_EQ(x.id, y.id);
            EXPECT_EQ(x.embedding, y.embedding);
            EXPECT_EQ(x.intrinsicQuality, y.intrinsicQuality);
            EXPECT_EQ(x.durationSeconds, y.durationSeconds);
            EXPECT_EQ(x.primaryTopic, y.primaryTopic);
            EXPECT_EQ(x.createdAt, y.createdAt);
        }
    }
}
