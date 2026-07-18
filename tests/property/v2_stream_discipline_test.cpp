#include "rr/simulation/dataset_generator.hpp"

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/archetype_config.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/simulation/hidden/hidden_reel_state.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/modality_space.hpp"
#include "rr/simulation/user_augmenter_v2.hpp" // userTraitsV2 ranges

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace rr;

// Phase 13 core compatibility proof (D17/D19): gate-off output is byte-identical V1 with ZERO V2
// draws, and gate-on leaves every V1 field/stream untouched (new draws come only from the new
// "archetypes"/"reels-v2"/"users-v2" streams). The domain structs have no operator==, so the
// comparison is done field-by-field below; the helpers are split explicitly into compareV1Fields /
// compareV2Fields so the split DOUBLES AS EXECUTABLE DOCUMENTATION of which fields are V1 vs V2.
//
// Package B owns the user side; several V2 fields compared here (reel attributes, modality spaces,
// hidden reel states) belong to package A and are all default/empty against A's Phase-13 STUB, so
// the tests touching them pass here in their default form and are re-verified with A's real output
// at integration. The user-side V2 assertions are exercised for real (augmentUsersV2 runs).

namespace {

constexpr uint32_t kNumSeeds = 24; // >= 20 (D17 property-test requirement)

SimulationConfig smallCfg() {
    SimulationConfig c;
    c.users = 40;
    c.reels = 60;
    c.creators = 20;
    c.topics = 8;
    c.dimensions = 16;
    return c;
}

RealismConfig onCfg() {
    RealismConfig r;
    r.contentV2 = true;
    r.languages = 8;
    return r;
}

// A trait is "in range" within a small float-rounding tolerance (see user_augmenter_v2_test.cpp).
constexpr double kTol = 1e-4;
::testing::AssertionResult inRange(const char *name, double v, double lo, double hi) {
    if (v >= lo - kTol && v <= hi + kTol) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << name << " = " << v << " outside [" << lo << ", " << hi << "]";
}

// ===========================================================================================
//  V1 field comparators — every field that existed before Phase 13 (the D17 stability surface)
// ===========================================================================================

void compareTopics(const std::vector<Topic> &a, const std::vector<Topic> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].id.value, b[i].id.value);
        EXPECT_EQ(a[i].centre, b[i].centre);
    }
}

void compareCreators(const std::vector<Creator> &a, const std::vector<Creator> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].id.value, b[i].id.value);
        EXPECT_EQ(a[i].styleEmbedding, b[i].styleEmbedding);
        EXPECT_EQ(a[i].topicSpecialties, b[i].topicSpecialties);
        EXPECT_EQ(a[i].baseQuality, b[i].baseQuality);
    }
}

void compareReelsV1(const std::vector<Reel> &a, const std::vector<Reel> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].id.value, b[i].id.value);
        EXPECT_EQ(a[i].creatorId.value, b[i].creatorId.value);
        EXPECT_EQ(a[i].embedding, b[i].embedding);
        EXPECT_EQ(a[i].intrinsicQuality, b[i].intrinsicQuality);
        EXPECT_EQ(a[i].freshnessScore, b[i].freshnessScore);
        EXPECT_EQ(a[i].durationSeconds, b[i].durationSeconds);
        EXPECT_EQ(a[i].primaryTopic.value, b[i].primaryTopic.value);
        EXPECT_EQ(a[i].secondaryTopics, b[i].secondaryTopics);
        EXPECT_EQ(a[i].createdAt, b[i].createdAt);
        EXPECT_EQ(a[i].impressionCount, b[i].impressionCount);
        EXPECT_EQ(a[i].completionCount, b[i].completionCount);
        EXPECT_EQ(a[i].likeCount, b[i].likeCount);
        EXPECT_EQ(a[i].shareCount, b[i].shareCount);
        EXPECT_EQ(a[i].skipCount, b[i].skipCount);
        EXPECT_EQ(a[i].active, b[i].active);
        EXPECT_EQ(a[i].trendingEngagement, b[i].trendingEngagement);
        EXPECT_EQ(a[i].trendingImpressions, b[i].trendingImpressions);
        EXPECT_EQ(a[i].trendingUpdatedAt, b[i].trendingUpdatedAt);
    }
}

void compareUsers(const std::vector<User> &a, const std::vector<User> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].id.value, b[i].id.value);
        EXPECT_EQ(a[i].estimatedPreference, b[i].estimatedPreference);
        EXPECT_EQ(a[i].longTermPreference, b[i].longTermPreference);
        EXPECT_EQ(a[i].sessionPreference, b[i].sessionPreference);
        EXPECT_EQ(a[i].totalInteractions, b[i].totalInteractions);
        EXPECT_EQ(a[i].currentSessionLength, b[i].currentSessionLength);
        EXPECT_EQ(a[i].seenReels.size(), b[i].seenReels.size());
        EXPECT_EQ(a[i].creatorAffinity.size(), b[i].creatorAffinity.size());
        EXPECT_EQ(a[i].recentInteractions.size(), b[i].recentInteractions.size());
    }
}

void compareHiddenV1(const std::vector<HiddenUserState> &a, const std::vector<HiddenUserState> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].userId.value, b[i].userId.value);
        EXPECT_EQ(a[i].hiddenPreference, b[i].hiddenPreference);
        EXPECT_EQ(a[i].preferredTopics, b[i].preferredTopics);
        EXPECT_EQ(a[i].preferenceConcentration, b[i].preferenceConcentration);
        EXPECT_EQ(a[i].exploreWillingness, b[i].exploreWillingness);
        EXPECT_EQ(a[i].avgSessionLength, b[i].avgSessionLength);
        EXPECT_EQ(a[i].likePropensity, b[i].likePropensity);
        EXPECT_EQ(a[i].sharePropensity, b[i].sharePropensity);
        EXPECT_EQ(a[i].durationTolerance, b[i].durationTolerance);
        EXPECT_EQ(a[i].preferenceStability, b[i].preferenceStability);
    }
}

void compareV1Fields(const GeneratedDataset &a, const GeneratedDataset &b) {
    compareTopics(a.topics, b.topics);
    compareCreators(a.creators, b.creators);
    compareReelsV1(a.reels, b.reels);
    compareUsers(a.users, b.users);
    compareHiddenV1(a.hiddenStates, b.hiddenStates);
}

// ===========================================================================================
//  V2 field comparators — everything Phase 13 adds
// ===========================================================================================

// User-side V2 channels/traits (package B). Isolated as its own helper because the realism-param
// independence tests compare ONLY the user side.
void compareHiddenV2(const std::vector<HiddenUserState> &a, const std::vector<HiddenUserState> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].visualPreference, b[i].visualPreference);
        EXPECT_EQ(a[i].musicPreference, b[i].musicPreference);
        EXPECT_EQ(a[i].emotionalPreference, b[i].emotionalPreference);
        EXPECT_EQ(a[i].usefulnessPreference, b[i].usefulnessPreference);
        EXPECT_EQ(a[i].humourPreference, b[i].humourPreference);
        EXPECT_EQ(a[i].controversyTolerance, b[i].controversyTolerance);
        EXPECT_EQ(a[i].noveltySeeking, b[i].noveltySeeking);
        EXPECT_EQ(a[i].clickbaitSusceptibility, b[i].clickbaitSusceptibility);
        EXPECT_EQ(a[i].informationTolerance, b[i].informationTolerance);
        EXPECT_EQ(a[i].primaryLanguage.value, b[i].primaryLanguage.value);
        EXPECT_EQ(a[i].languageMismatchTolerance, b[i].languageMismatchTolerance);
        EXPECT_EQ(a[i].repetitionTolerance, b[i].repetitionTolerance);
        EXPECT_EQ(a[i].noveltyTolerance, b[i].noveltyTolerance);
        EXPECT_EQ(a[i].creatorLoyalty, b[i].creatorLoyalty);
        EXPECT_EQ(a[i].habitStrength, b[i].habitStrength);
        EXPECT_EQ(a[i].platformTrust, b[i].platformTrust);
        EXPECT_EQ(a[i].baselineDailyUsage, b[i].baselineDailyUsage);
        EXPECT_EQ(a[i].preferencePlasticity, b[i].preferencePlasticity);
    }
}

// Reel-side V2 attributes (package A). Compared for the determinism proof; default/empty vs A's
// stub, populated for real at integration.
void compareReelsV2(const std::vector<Reel> &a, const std::vector<Reel> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].visualStyleEmbedding, b[i].visualStyleEmbedding);
        EXPECT_EQ(a[i].musicEmbedding, b[i].musicEmbedding);
        EXPECT_EQ(a[i].emotionalToneEmbedding, b[i].emotionalToneEmbedding);
        EXPECT_EQ(a[i].usefulness, b[i].usefulness);
        EXPECT_EQ(a[i].humour, b[i].humour);
        EXPECT_EQ(a[i].novelty, b[i].novelty);
        EXPECT_EQ(a[i].productionQuality, b[i].productionQuality);
        EXPECT_EQ(a[i].controversy, b[i].controversy);
        EXPECT_EQ(a[i].clickbaitStrength, b[i].clickbaitStrength);
        EXPECT_EQ(a[i].informationDensity, b[i].informationDensity);
        EXPECT_EQ(a[i].emotionalIntensity, b[i].emotionalIntensity);
        EXPECT_EQ(a[i].language.value, b[i].language.value);
    }
}

void compareModalitySpaces(const ModalitySpaces &a, const ModalitySpaces &b) {
    EXPECT_EQ(a.visualCentres, b.visualCentres);
    EXPECT_EQ(a.musicCentres, b.musicCentres);
    EXPECT_EQ(a.emotionalCentres, b.emotionalCentres);
}

void compareHiddenReelStates(const std::vector<HiddenReelState> &a,
                             const std::vector<HiddenReelState> &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].reelId.value, b[i].reelId.value);
        EXPECT_EQ(a[i].archetypeIndex, b[i].archetypeIndex);
        EXPECT_EQ(a[i].satisfactionBias, b[i].satisfactionBias);
        EXPECT_EQ(a[i].regretBias, b[i].regretBias);
        EXPECT_EQ(a[i].openingHook, b[i].openingHook);
        EXPECT_EQ(a[i].retentionDecay, b[i].retentionDecay);
        EXPECT_EQ(a[i].nicheCohortCentre, b[i].nicheCohortCentre);
        EXPECT_EQ(a[i].nicheCohortWidth, b[i].nicheCohortWidth);
        EXPECT_EQ(a[i].comfortReturnBonus, b[i].comfortReturnBonus);
    }
}

void compareV2Fields(const GeneratedDataset &a, const GeneratedDataset &b) {
    compareHiddenV2(a.hiddenStates, b.hiddenStates);
    compareReelsV2(a.reels, b.reels);
    compareModalitySpaces(a.modalitySpaces, b.modalitySpaces);
    compareHiddenReelStates(a.hiddenReelStates, b.hiddenReelStates);
}

// Every Phase-13 field is default-initialized (the observable consequence of ZERO V2 draws under
// gate-off: augmenters are never called, so structures keep their defaults).
void assertV2FieldsDefault(const GeneratedDataset &ds) {
    for (const auto &h : ds.hiddenStates) {
        EXPECT_TRUE(h.visualPreference.empty());
        EXPECT_TRUE(h.musicPreference.empty());
        EXPECT_TRUE(h.emotionalPreference.empty());
        EXPECT_EQ(h.usefulnessPreference, 0.0f);
        EXPECT_EQ(h.humourPreference, 0.0f);
        EXPECT_EQ(h.controversyTolerance, 0.0f);
        EXPECT_EQ(h.noveltySeeking, 0.0f);
        EXPECT_EQ(h.clickbaitSusceptibility, 0.0f);
        EXPECT_EQ(h.informationTolerance, 0.0f);
        EXPECT_EQ(h.primaryLanguage.value, 0u);
        EXPECT_EQ(h.languageMismatchTolerance, 0.0f);
        EXPECT_EQ(h.repetitionTolerance, 0.0f);
        EXPECT_EQ(h.noveltyTolerance, 0.0f);
        EXPECT_EQ(h.creatorLoyalty, 0.0f);
        EXPECT_EQ(h.habitStrength, 0.0f);
        EXPECT_EQ(h.platformTrust, 0.0f);
        EXPECT_EQ(h.baselineDailyUsage, 0.0f);
        EXPECT_EQ(h.preferencePlasticity, 0.0f);
    }
    for (const auto &r : ds.reels) {
        EXPECT_TRUE(r.visualStyleEmbedding.empty());
        EXPECT_TRUE(r.musicEmbedding.empty());
        EXPECT_TRUE(r.emotionalToneEmbedding.empty());
        EXPECT_EQ(r.usefulness, 0.0f);
        EXPECT_EQ(r.humour, 0.0f);
        EXPECT_EQ(r.novelty, 0.0f);
        EXPECT_EQ(r.productionQuality, 0.0f);
        EXPECT_EQ(r.controversy, 0.0f);
        EXPECT_EQ(r.clickbaitStrength, 0.0f);
        EXPECT_EQ(r.informationDensity, 0.0f);
        EXPECT_EQ(r.emotionalIntensity, 0.0f);
        EXPECT_EQ(r.language.value, 0u);
    }
    EXPECT_TRUE(ds.modalitySpaces.visualCentres.empty());
    EXPECT_TRUE(ds.modalitySpaces.musicCentres.empty());
    EXPECT_TRUE(ds.modalitySpaces.emotionalCentres.empty());
    EXPECT_TRUE(ds.hiddenReelStates.empty());
}

// ===========================================================================================
//  (i) GATE-OFF IDENTITY: 2-arg overload == 3-arg gate-off, and every V2 field default.
// ===========================================================================================
TEST(V2StreamDiscipline, GateOffByteIdentityAndZeroV2Draws) {
    const SimulationConfig c = smallCfg();
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset twoArg = generateDataset(c, seed);
        GeneratedDataset gateOff = generateDataset(c, RealismConfig{}, seed);
        compareV1Fields(twoArg, gateOff);
        compareV2Fields(twoArg, gateOff);
        assertV2FieldsDefault(twoArg);
        assertV2FieldsDefault(gateOff);
    }
}

// ===========================================================================================
//  (ii) GATE-ON V1 STABILITY: gate-on vs gate-off — every V1 field byte-identical.
// ===========================================================================================
TEST(V2StreamDiscipline, GateOnLeavesV1FieldsByteIdentical) {
    const SimulationConfig c = smallCfg();
    const RealismConfig on = onCfg();
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset gateOff = generateDataset(c, RealismConfig{}, seed);
        GeneratedDataset gateOn = generateDataset(c, on, seed);
        compareV1Fields(gateOff, gateOn);
        // Sanity: the user-side V2 channels ARE populated under gate-on (augmentUsersV2 ran).
        // With A's stub the modality spaces are empty, so the modality preferences come from the
        // documented deterministic fallback and are still non-empty unit vectors.
        ASSERT_FALSE(gateOn.hiddenStates.empty());
        EXPECT_FALSE(gateOn.hiddenStates[0].visualPreference.empty());
    }
}

// ===========================================================================================
//  (iii) GATE-ON DETERMINISM: same seed twice with gate on => identical everything, incl. V2.
// ===========================================================================================
TEST(V2StreamDiscipline, GateOnSameSeedIsFullyDeterministic) {
    const SimulationConfig c = smallCfg();
    const RealismConfig on = onCfg();
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset a = generateDataset(c, on, seed);
        GeneratedDataset b = generateDataset(c, on, seed);
        compareV1Fields(a, b);
        compareV2Fields(a, b);
    }
}

// ===========================================================================================
//  (iv) REALISM-PARAM INDEPENDENCE: user V2 fields live on "users-v2", untouched by the archetype
//       catalog ("archetypes"/"reels-v2") or the reel count.
// ===========================================================================================

// Changing the archetype catalog (weights AND non-weight params) leaves every USER-side V2 field
// identical. NOTE: against package A's stub augmentReelsV2 is a no-op, so this passes trivially; it
// is re-verified at integration where the catalog actually drives the "archetypes"/"reels-v2"
// draws (and A's added distribution parameters are also varied).
TEST(V2StreamDiscipline, ChangingArchetypeCatalogLeavesUserV2Identical) {
    const SimulationConfig c = smallCfg();
    RealismConfig base = onCfg();
    RealismConfig alt = onCfg();
    alt.archetypes = {
        ArchetypeSpec{"alpha", 2.0},
        ArchetypeSpec{"beta", 0.5},
        ArchetypeSpec{"gamma", 3.0},
    };
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset dsBase = generateDataset(c, base, seed);
        GeneratedDataset dsAlt = generateDataset(c, alt, seed);
        compareHiddenV2(dsBase.hiddenStates, dsAlt.hiddenStates);
    }
}

// Changing simulation.reels (500 -> 800 style) leaves every USER V2 field identical: the modality
// centres are drawn BEFORE any per-reel draw (package A's pinned contract), and the user draws live
// on the independent "users-v2" stream. NOTE: trivial under A's stub (empty modality spaces =>
// fallback path, independent of reels); re-verify at integration.
TEST(V2StreamDiscipline, ChangingReelCountLeavesUserV2Identical) {
    SimulationConfig cFew = smallCfg();
    SimulationConfig cMany = smallCfg();
    cMany.reels = cFew.reels * 2; // more reels, SAME users
    const RealismConfig on = onCfg();
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset dsFew = generateDataset(cFew, on, seed);
        GeneratedDataset dsMany = generateDataset(cMany, on, seed);
        ASSERT_EQ(dsFew.hiddenStates.size(), dsMany.hiddenStates.size());
        compareHiddenV2(dsFew.hiddenStates, dsMany.hiddenStates);
    }
}

// ===========================================================================================
//  (v) RANGES: every V2 scalar in its userTraitsV2 range, modality preferences unit-norm,
//      primaryLanguage < realism.languages — across all seeds.
// ===========================================================================================
TEST(V2StreamDiscipline, GateOnValuesWithinDocumentedRanges) {
    const SimulationConfig c = smallCfg();
    const RealismConfig on = onCfg();
    for (uint64_t seed = 1; seed <= kNumSeeds; ++seed) {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        GeneratedDataset ds = generateDataset(c, on, seed);
        ASSERT_EQ(ds.hiddenStates.size(), c.users);
        for (const auto &h : ds.hiddenStates) {
            EXPECT_TRUE(inRange("usefulnessPreference", h.usefulnessPreference,
                                userTraitsV2::kUsefulnessPreferenceLo,
                                userTraitsV2::kUsefulnessPreferenceHi));
            EXPECT_TRUE(inRange("humourPreference", h.humourPreference,
                                userTraitsV2::kHumourPreferenceLo,
                                userTraitsV2::kHumourPreferenceHi));
            EXPECT_TRUE(inRange("controversyTolerance", h.controversyTolerance,
                                userTraitsV2::kControversyToleranceLo,
                                userTraitsV2::kControversyToleranceHi));
            EXPECT_TRUE(inRange("noveltySeeking", h.noveltySeeking, userTraitsV2::kNoveltySeekingLo,
                                userTraitsV2::kNoveltySeekingHi));
            EXPECT_TRUE(inRange("clickbaitSusceptibility", h.clickbaitSusceptibility,
                                userTraitsV2::kClickbaitSusceptibilityLo,
                                userTraitsV2::kClickbaitSusceptibilityHi));
            EXPECT_TRUE(inRange("informationTolerance", h.informationTolerance,
                                userTraitsV2::kInformationToleranceLo,
                                userTraitsV2::kInformationToleranceHi));
            EXPECT_TRUE(inRange("languageMismatchTolerance", h.languageMismatchTolerance,
                                userTraitsV2::kLanguageMismatchToleranceLo,
                                userTraitsV2::kLanguageMismatchToleranceHi));
            EXPECT_TRUE(inRange("repetitionTolerance", h.repetitionTolerance,
                                userTraitsV2::kRepetitionToleranceLo,
                                userTraitsV2::kRepetitionToleranceHi));
            EXPECT_TRUE(inRange("noveltyTolerance", h.noveltyTolerance,
                                userTraitsV2::kNoveltyToleranceLo,
                                userTraitsV2::kNoveltyToleranceHi));
            EXPECT_TRUE(inRange("creatorLoyalty", h.creatorLoyalty, userTraitsV2::kCreatorLoyaltyLo,
                                userTraitsV2::kCreatorLoyaltyHi));
            EXPECT_TRUE(inRange("habitStrength", h.habitStrength, userTraitsV2::kHabitStrengthLo,
                                userTraitsV2::kHabitStrengthHi));
            EXPECT_TRUE(inRange("platformTrust", h.platformTrust, userTraitsV2::kPlatformTrustLo,
                                userTraitsV2::kPlatformTrustHi));
            EXPECT_TRUE(inRange("baselineDailyUsage", h.baselineDailyUsage,
                                userTraitsV2::kBaselineDailyUsageLo,
                                userTraitsV2::kBaselineDailyUsageHi));
            EXPECT_TRUE(inRange("preferencePlasticity", h.preferencePlasticity,
                                userTraitsV2::kPreferencePlasticityLo,
                                userTraitsV2::kPreferencePlasticityHi));
            EXPECT_TRUE(isValid(h.visualPreference));
            EXPECT_TRUE(isValid(h.musicPreference));
            EXPECT_TRUE(isValid(h.emotionalPreference));
            EXPECT_LT(h.primaryLanguage.value, on.languages);
        }
    }
}

} // namespace
