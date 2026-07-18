#include "rr/simulation/dataset_generator.hpp"

#include "rr/core/embedding.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp" // splitmix64
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/user_augmenter_v2.hpp" // userTraitsV2 ranges

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <vector>

using namespace rr;

// Phase 13 exit criterion 1 (D17): the FULL V2 dataset (100k reels / 10k users / 5k creators /
// 32 topics / dim 64) generates deterministically with the content_v2 gate ON, in sane time, with
// every user-side V2 field valid at scale. Determinism is checked via a fingerprint of all user V2
// fields (rather than holding two full datasets in memory): generate, fingerprint, free,
// regenerate, fingerprint, compare.
//
// This exercises package B's augmentUsersV2 at scale. Against package A's Phase-13 stub the reel
// modality spaces are empty, so the user modality preferences come from the documented
// deterministic fallback; the fingerprint therefore proves package B's determinism regardless of A.
// The full gate-on dataset (with A's real reel attributes / modality spaces) is re-verified at
// integration.

namespace {

constexpr double kTol = 1e-4;

// Order-sensitive fingerprint over every user-side V2 field, folded through the project's
// splitmix64 finalizer. Two runs with the same seed must produce the same value.
uint64_t fingerprintUserV2(const std::vector<HiddenUserState> &hs) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    auto mixU = [&](uint64_t v) { h = splitmix64(h ^ v); };
    auto mixF = [&](float f) { mixU(static_cast<uint64_t>(std::bit_cast<uint32_t>(f))); };
    auto mixE = [&](const Embedding &e) {
        mixU(e.size());
        for (float f : e) {
            mixF(f);
        }
    };
    for (const auto &s : hs) {
        mixU(s.userId.value);
        mixE(s.visualPreference);
        mixE(s.musicPreference);
        mixE(s.emotionalPreference);
        mixF(s.usefulnessPreference);
        mixF(s.humourPreference);
        mixF(s.controversyTolerance);
        mixF(s.noveltySeeking);
        mixF(s.clickbaitSusceptibility);
        mixF(s.informationTolerance);
        mixU(s.primaryLanguage.value);
        mixF(s.languageMismatchTolerance);
        mixF(s.repetitionTolerance);
        mixF(s.noveltyTolerance);
        mixF(s.creatorLoyalty);
        mixF(s.habitStrength);
        mixF(s.platformTrust);
        mixF(s.baselineDailyUsage);
        mixF(s.preferencePlasticity);
    }
    return h;
}

bool le(float v, double hi) { return v <= hi + kTol; }
bool ge(float v, double lo) { return v >= lo - kTol; }

::testing::AssertionResult scalarsInRange(const HiddenUserState &h) {
    if (ge(h.usefulnessPreference, userTraitsV2::kUsefulnessPreferenceLo) &&
        le(h.usefulnessPreference, userTraitsV2::kUsefulnessPreferenceHi) &&
        ge(h.humourPreference, userTraitsV2::kHumourPreferenceLo) &&
        le(h.humourPreference, userTraitsV2::kHumourPreferenceHi) &&
        ge(h.controversyTolerance, userTraitsV2::kControversyToleranceLo) &&
        le(h.controversyTolerance, userTraitsV2::kControversyToleranceHi) &&
        ge(h.noveltySeeking, userTraitsV2::kNoveltySeekingLo) &&
        le(h.noveltySeeking, userTraitsV2::kNoveltySeekingHi) &&
        ge(h.clickbaitSusceptibility, userTraitsV2::kClickbaitSusceptibilityLo) &&
        le(h.clickbaitSusceptibility, userTraitsV2::kClickbaitSusceptibilityHi) &&
        ge(h.informationTolerance, userTraitsV2::kInformationToleranceLo) &&
        le(h.informationTolerance, userTraitsV2::kInformationToleranceHi) &&
        ge(h.languageMismatchTolerance, userTraitsV2::kLanguageMismatchToleranceLo) &&
        le(h.languageMismatchTolerance, userTraitsV2::kLanguageMismatchToleranceHi) &&
        ge(h.repetitionTolerance, userTraitsV2::kRepetitionToleranceLo) &&
        le(h.repetitionTolerance, userTraitsV2::kRepetitionToleranceHi) &&
        ge(h.noveltyTolerance, userTraitsV2::kNoveltyToleranceLo) &&
        le(h.noveltyTolerance, userTraitsV2::kNoveltyToleranceHi) &&
        ge(h.creatorLoyalty, userTraitsV2::kCreatorLoyaltyLo) &&
        le(h.creatorLoyalty, userTraitsV2::kCreatorLoyaltyHi) &&
        ge(h.habitStrength, userTraitsV2::kHabitStrengthLo) &&
        le(h.habitStrength, userTraitsV2::kHabitStrengthHi) &&
        ge(h.platformTrust, userTraitsV2::kPlatformTrustLo) &&
        le(h.platformTrust, userTraitsV2::kPlatformTrustHi) &&
        ge(h.baselineDailyUsage, userTraitsV2::kBaselineDailyUsageLo) &&
        le(h.baselineDailyUsage, userTraitsV2::kBaselineDailyUsageHi) &&
        ge(h.preferencePlasticity, userTraitsV2::kPreferencePlasticityLo) &&
        le(h.preferencePlasticity, userTraitsV2::kPreferencePlasticityHi)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << "user " << h.userId.value << " has a V2 scalar out of range";
}

SimulationConfig fullScaleCfg() {
    // Exit criterion 1 sizes. This is a Debug-build test; the double-generation below stays well
    // under the ~60s budget locally (the V1 reel generator dominates, package B's augmentation is
    // ~2M cheap draws over 10k users). If a slower environment exceeds the budget, drop reels to
    // 50k — the determinism/validity proof is unaffected by the exact count.
    SimulationConfig c;
    c.users = 10000;
    c.reels = 100000;
    c.creators = 5000;
    c.topics = 32;
    c.dimensions = 64;
    return c;
}

} // namespace

TEST(V2DatasetGenerationIntegration, FullScaleGateOnDeterministicAndValid) {
    const SimulationConfig cfg = fullScaleCfg();
    RealismConfig realism;
    realism.contentV2 = true; // languages defaults to 8
    const uint64_t seed = 20260718;

    uint64_t fp1 = 0;
    {
        GeneratedDataset ds = generateDataset(cfg, realism, seed);
        ASSERT_EQ(ds.hiddenStates.size(), cfg.users);
        ASSERT_EQ(ds.users.size(), cfg.users);
        ASSERT_EQ(ds.reels.size(), cfg.reels);

        // Validity at scale: every user's three modality preferences are valid unit vectors of the
        // configured dimension, every V2 scalar is within its documented range, and the primary
        // language is a valid id.
        for (const auto &h : ds.hiddenStates) {
            ASSERT_TRUE(isValid(h.visualPreference));
            ASSERT_TRUE(isValid(h.musicPreference));
            ASSERT_TRUE(isValid(h.emotionalPreference));
            ASSERT_EQ(h.visualPreference.size(), cfg.dimensions);
            ASSERT_EQ(h.musicPreference.size(), cfg.dimensions);
            ASSERT_EQ(h.emotionalPreference.size(), cfg.dimensions);
            ASSERT_TRUE(scalarsInRange(h));
            ASSERT_LT(h.primaryLanguage.value, realism.languages);
        }
        fp1 = fingerprintUserV2(ds.hiddenStates);
    } // ds freed here — peak memory stays at one dataset

    GeneratedDataset ds2 = generateDataset(cfg, realism, seed);
    const uint64_t fp2 = fingerprintUserV2(ds2.hiddenStates);
    EXPECT_EQ(fp1, fp2) << "full-scale gate-on generation is not deterministic across runs";
}
