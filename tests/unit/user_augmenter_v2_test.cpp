#include "rr/simulation/user_augmenter_v2.hpp"

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/modality_space.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace rr;

namespace {

// A trait is "in range" if it lands within [lo, hi] allowing a small float-rounding tolerance
// (Rng::uniform draws a double in [lo, hi); casting to float can nudge it a hair past a bound).
// Mirrors the userTraits precedent in user_generator_test.cpp.
constexpr double kTol = 1e-4;
::testing::AssertionResult inRange(const char *name, double v, double lo, double hi) {
    if (v >= lo - kTol && v <= hi + kTol) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << name << " = " << v << " outside [" << lo << ", " << hi << "]";
}

// N L2-normalized random centres of `dim` dimensions (self-contained; does not depend on package
// A's generateModalitySpaces — lets the unit tests exercise the REAL centre-blend path directly).
std::vector<Embedding> makeCentres(uint32_t count, uint32_t dim, uint64_t seed) {
    Rng rng(seed);
    std::vector<Embedding> centres;
    centres.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Embedding c(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            c[d] = static_cast<float>(rng.gaussian());
        }
        normalize(c);
        centres.push_back(std::move(c));
    }
    return centres;
}

ModalitySpaces makeSpaces(uint32_t visual, uint32_t music, uint32_t emotional, uint32_t dim,
                          uint64_t seed) {
    ModalitySpaces s;
    s.visualCentres = makeCentres(visual, dim, seed + 1);
    s.musicCentres = makeCentres(music, dim, seed + 2);
    s.emotionalCentres = makeCentres(emotional, dim, seed + 3);
    return s;
}

std::vector<HiddenUserState> makeStates(uint32_t n) {
    std::vector<HiddenUserState> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i].userId = UserId{i};
    }
    return v;
}

SimulationConfig cfg(uint32_t users, uint32_t dim) {
    SimulationConfig c;
    c.users = users;
    c.dimensions = dim;
    return c;
}

RealismConfig realismOn(uint32_t languages) {
    RealismConfig r;
    r.contentV2 = true;
    r.languages = languages;
    return r;
}

// Assert every V2 scalar of `h` falls in its userTraitsV2 range and both bounds are respected.
void expectScalarsInRange(const HiddenUserState &h) {
    EXPECT_TRUE(inRange("usefulnessPreference", h.usefulnessPreference,
                        userTraitsV2::kUsefulnessPreferenceLo,
                        userTraitsV2::kUsefulnessPreferenceHi));
    EXPECT_TRUE(inRange("humourPreference", h.humourPreference, userTraitsV2::kHumourPreferenceLo,
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
    EXPECT_TRUE(inRange("noveltyTolerance", h.noveltyTolerance, userTraitsV2::kNoveltyToleranceLo,
                        userTraitsV2::kNoveltyToleranceHi));
    EXPECT_TRUE(inRange("creatorLoyalty", h.creatorLoyalty, userTraitsV2::kCreatorLoyaltyLo,
                        userTraitsV2::kCreatorLoyaltyHi));
    EXPECT_TRUE(inRange("habitStrength", h.habitStrength, userTraitsV2::kHabitStrengthLo,
                        userTraitsV2::kHabitStrengthHi));
    EXPECT_TRUE(inRange("platformTrust", h.platformTrust, userTraitsV2::kPlatformTrustLo,
                        userTraitsV2::kPlatformTrustHi));
    EXPECT_TRUE(inRange("baselineDailyUsage", h.baselineDailyUsage,
                        userTraitsV2::kBaselineDailyUsageLo, userTraitsV2::kBaselineDailyUsageHi));
    EXPECT_TRUE(inRange("preferencePlasticity", h.preferencePlasticity,
                        userTraitsV2::kPreferencePlasticityLo,
                        userTraitsV2::kPreferencePlasticityHi));
}

// Field-by-field equality of the V2 channels/traits (the domain struct has no operator==).
void expectSameV2(const HiddenUserState &a, const HiddenUserState &b) {
    EXPECT_EQ(a.visualPreference, b.visualPreference);
    EXPECT_EQ(a.musicPreference, b.musicPreference);
    EXPECT_EQ(a.emotionalPreference, b.emotionalPreference);
    EXPECT_EQ(a.usefulnessPreference, b.usefulnessPreference);
    EXPECT_EQ(a.humourPreference, b.humourPreference);
    EXPECT_EQ(a.controversyTolerance, b.controversyTolerance);
    EXPECT_EQ(a.noveltySeeking, b.noveltySeeking);
    EXPECT_EQ(a.clickbaitSusceptibility, b.clickbaitSusceptibility);
    EXPECT_EQ(a.informationTolerance, b.informationTolerance);
    EXPECT_EQ(a.primaryLanguage.value, b.primaryLanguage.value);
    EXPECT_EQ(a.languageMismatchTolerance, b.languageMismatchTolerance);
    EXPECT_EQ(a.repetitionTolerance, b.repetitionTolerance);
    EXPECT_EQ(a.noveltyTolerance, b.noveltyTolerance);
    EXPECT_EQ(a.creatorLoyalty, b.creatorLoyalty);
    EXPECT_EQ(a.habitStrength, b.habitStrength);
    EXPECT_EQ(a.platformTrust, b.platformTrust);
    EXPECT_EQ(a.baselineDailyUsage, b.baselineDailyUsage);
    EXPECT_EQ(a.preferencePlasticity, b.preferencePlasticity);
}

} // namespace

// Every generated V2 scalar within its documented range; every modality preference a valid unit
// vector; primaryLanguage < languages. Exercises the REAL centre-blend path (non-empty spaces).
TEST(UserAugmenterV2Test, RangesAndNormalizationRealSpaces) {
    const uint32_t dim = 32;
    const uint32_t n = 300;
    const uint32_t languages = 8;
    auto spaces = makeSpaces(5, 4, 6, dim, 100);
    auto states = makeStates(n);
    Rng rng(42);
    augmentUsersV2(states, spaces, cfg(n, dim), realismOn(languages), rng);

    for (const auto &h : states) {
        expectScalarsInRange(h);
        EXPECT_TRUE(isValid(h.visualPreference)) << "visualPreference not a unit vector";
        EXPECT_TRUE(isValid(h.musicPreference)) << "musicPreference not a unit vector";
        EXPECT_TRUE(isValid(h.emotionalPreference)) << "emotionalPreference not a unit vector";
        EXPECT_EQ(h.visualPreference.size(), dim);
        EXPECT_LT(h.primaryLanguage.value, languages);
    }
}

// Same rng seed twice => byte-identical V2 fields (determinism depends only on the pinned order).
TEST(UserAugmenterV2Test, Deterministic) {
    const uint32_t dim = 32;
    const uint32_t n = 100;
    auto spaces = makeSpaces(5, 4, 6, dim, 100);

    auto a = makeStates(n);
    Rng ra(7);
    augmentUsersV2(a, spaces, cfg(n, dim), realismOn(8), ra);

    auto b = makeStates(n);
    Rng rb(7);
    augmentUsersV2(b, spaces, cfg(n, dim), realismOn(8), rb);

    for (uint32_t i = 0; i < n; ++i) {
        expectSameV2(a[i], b[i]);
    }
}

// Different seeds => the draws actually happen (results differ).
TEST(UserAugmenterV2Test, DifferentSeedsDiffer) {
    const uint32_t dim = 32;
    const uint32_t n = 50;
    auto spaces = makeSpaces(5, 4, 6, dim, 100);

    auto a = makeStates(n);
    Rng ra(1);
    augmentUsersV2(a, spaces, cfg(n, dim), realismOn(8), ra);
    auto b = makeStates(n);
    Rng rb(2);
    augmentUsersV2(b, spaces, cfg(n, dim), realismOn(8), rb);

    bool anyDiff = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (a[i].usefulnessPreference != b[i].usefulnessPreference ||
            a[i].visualPreference != b[i].visualPreference) {
            anyDiff = true;
            break;
        }
    }
    EXPECT_TRUE(anyDiff);
}

// The modality centres are genuinely CONSUMED: with the same rng seed but different centre VALUES
// (same counts, so identical count/selection/weight draws), the blended preference differs. This
// proves the real blend path reads the shared ModalitySpaces (not just noise).
TEST(UserAugmenterV2Test, ModalityCentresConsumed) {
    const uint32_t dim = 32;
    const uint32_t n = 20;
    auto spacesA = makeSpaces(5, 4, 6, dim, 100);
    auto spacesB = makeSpaces(5, 4, 6, dim, 999); // same sizes, different centre values

    auto a = makeStates(n);
    Rng ra(7);
    augmentUsersV2(a, spacesA, cfg(n, dim), realismOn(8), ra);
    auto b = makeStates(n);
    Rng rb(7);
    augmentUsersV2(b, spacesB, cfg(n, dim), realismOn(8), rb);

    bool anyDiff = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (a[i].visualPreference != b[i].visualPreference) {
            anyDiff = true;
            break;
        }
    }
    EXPECT_TRUE(anyDiff) << "modality centres did not affect the preference — blend path not taken";
}

// Empty ModalitySpaces (package A's stub, or a degenerate config): the documented deterministic
// fallback still produces valid unit vectors, deterministically, with scalars in range.
TEST(UserAugmenterV2Test, EmptyModalitySpacesFallback) {
    const uint32_t dim = 32;
    const uint32_t n = 100;
    ModalitySpaces empty; // all three centre sets empty

    auto a = makeStates(n);
    Rng ra(7);
    augmentUsersV2(a, empty, cfg(n, dim), realismOn(8), ra);
    auto b = makeStates(n);
    Rng rb(7);
    augmentUsersV2(b, empty, cfg(n, dim), realismOn(8), rb);

    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_TRUE(isValid(a[i].visualPreference)) << "fallback visualPreference not unit-norm";
        EXPECT_TRUE(isValid(a[i].musicPreference)) << "fallback musicPreference not unit-norm";
        EXPECT_TRUE(isValid(a[i].emotionalPreference))
            << "fallback emotionalPreference not unit-norm";
        EXPECT_EQ(a[i].visualPreference.size(), dim);
        expectScalarsInRange(a[i]);
        expectSameV2(a[i], b[i]); // deterministic fallback
    }
}

// V1 HiddenUserState fields are NOT touched by augmentUsersV2 (D17/D18): pre-populated V1 values
// survive verbatim; only the V2 channels/traits are written.
TEST(UserAugmenterV2Test, V1FieldsUntouched) {
    const uint32_t dim = 16;
    auto spaces = makeSpaces(4, 4, 4, dim, 100);

    HiddenUserState seed;
    seed.userId = UserId{123};
    seed.hiddenPreference = Embedding(dim, 0.0f);
    seed.hiddenPreference[0] = 1.0f;
    seed.preferredTopics = {TopicId{3}, TopicId{7}};
    seed.preferenceConcentration = 2.5f;
    seed.exploreWillingness = 0.3f;
    seed.avgSessionLength = 22.0f;
    seed.likePropensity = 0.11f;
    seed.sharePropensity = 0.04f;
    seed.durationTolerance = 0.6f;
    seed.preferenceStability = 0.8f;

    std::vector<HiddenUserState> states{seed};
    Rng rng(42);
    augmentUsersV2(states, spaces, cfg(1, dim), realismOn(8), rng);

    const HiddenUserState &h = states[0];
    EXPECT_EQ(h.userId.value, 123u);
    EXPECT_EQ(h.hiddenPreference, seed.hiddenPreference);
    EXPECT_EQ(h.preferredTopics, seed.preferredTopics);
    EXPECT_EQ(h.preferenceConcentration, 2.5f);
    EXPECT_EQ(h.exploreWillingness, 0.3f);
    EXPECT_EQ(h.avgSessionLength, 22.0f);
    EXPECT_EQ(h.likePropensity, 0.11f);
    EXPECT_EQ(h.sharePropensity, 0.04f);
    EXPECT_EQ(h.durationTolerance, 0.6f);
    EXPECT_EQ(h.preferenceStability, 0.8f);

    // ... and the V2 fields WERE written (sanity: not still default).
    EXPECT_EQ(h.visualPreference.size(), dim);
    EXPECT_TRUE(isValid(h.visualPreference));
}

// primaryLanguage is drawn from the skewed global distribution (rr::languageWeights): id 0 (the
// dominant language) is sampled strictly more often than the rarest id, and every id is in range.
TEST(UserAugmenterV2Test, LanguageDistributionSkew) {
    const uint32_t dim = 16;
    const uint32_t n = 5000;
    const uint32_t languages = 8;
    auto spaces = makeSpaces(4, 4, 4, dim, 100);
    auto states = makeStates(n);
    Rng rng(42);
    augmentUsersV2(states, spaces, cfg(n, dim), realismOn(languages), rng);

    std::vector<uint32_t> counts(languages, 0);
    for (const auto &h : states) {
        ASSERT_LT(h.primaryLanguage.value, languages);
        counts[h.primaryLanguage.value]++;
    }
    // Zipf(s=1): weight_0 = 8x weight_7, so id 0 must dominate id 7 comfortably over 5000 draws.
    EXPECT_GT(counts[0], counts[languages - 1]);
    EXPECT_GT(counts[0], 0u);
}

// languages == 1: every user gets language id 0; still deterministic and in-range.
TEST(UserAugmenterV2Test, SingleLanguage) {
    const uint32_t dim = 16;
    const uint32_t n = 30;
    auto spaces = makeSpaces(4, 4, 4, dim, 100);
    auto states = makeStates(n);
    Rng rng(42);
    augmentUsersV2(states, spaces, cfg(n, dim), realismOn(1), rng);
    for (const auto &h : states) {
        EXPECT_EQ(h.primaryLanguage.value, 0u);
    }
}
