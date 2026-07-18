#pragma once

#include <vector>

#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/modality_space.hpp"

namespace rr {

// Documented sampling ranges for the Realism V2 hidden user channels and traits (V2 TDD 4.2 + 5).
// This is the SINGLE SOURCE OF TRUTH (the userTraits precedent, user_generator.hpp): augmentUsersV2
// samples within these ranges and the tests verify every generated value falls inside them. Each
// scalar is [lo, hi] with lo inclusive; Rng::uniform draws a double in [lo, hi). Values are cast to
// float, so the tests allow a small rounding tolerance at the bounds.
namespace userTraitsV2 {

// --- (a) Per-modality preference embeddings (visual / music / emotional) ---------------------
// Each preference is an L2-normalized blend of kMinModalityCentres..kMaxModalityCentres DISTINCT
// centres drawn from the SAME shared ModalitySpaces the reels blend, plus per-dimension gaussian
// noise. Mirrors the generateUsers topic-mix machinery (user_generator.cpp): pick a centre count,
// select that many distinct centres via partial Fisher-Yates, weight each, add noise, normalize.
inline constexpr int kMinModalityCentres = 1;
inline constexpr int kMaxModalityCentres = 3;
// Blend weight per centre: base drawn from [kModalityWeightBaseLo, 1.0) (mirrors
// userTraits::kWeightBaseLo — the strictly-positive floor keeps every weight > 0).
inline constexpr double kModalityWeightBaseLo = 0.1;
// Std-dev of the per-dimension additive gaussian noise applied before normalization (mirrors
// userTraits::kNoiseScale).
inline constexpr double kModalityNoiseScale = 0.05;

// --- (b) Scalar content-value preferences and susceptibility traits (V2 TDD 4.2), all [0, 1] --
inline constexpr double kUsefulnessPreferenceLo = 0.0;
inline constexpr double kUsefulnessPreferenceHi = 1.0;
inline constexpr double kHumourPreferenceLo = 0.0;
inline constexpr double kHumourPreferenceHi = 1.0;
inline constexpr double kControversyToleranceLo = 0.0;
inline constexpr double kControversyToleranceHi = 1.0;
inline constexpr double kNoveltySeekingLo = 0.0;
inline constexpr double kNoveltySeekingHi = 1.0;
inline constexpr double kClickbaitSusceptibilityLo = 0.0;
inline constexpr double kClickbaitSusceptibilityHi = 1.0;
inline constexpr double kInformationToleranceLo = 0.0;
inline constexpr double kInformationToleranceHi = 1.0;

// --- (c) Language affinity -------------------------------------------------------------------
// Willingness to keep watching mismatched-language content (probability-like). [0, 1].
inline constexpr double kLanguageMismatchToleranceLo = 0.0;
inline constexpr double kLanguageMismatchToleranceHi = 1.0;

// --- (d) Forward traits: generated now, consumed by the NAMED later phase (V2 TDD 5) ---------
// P16-17 fatigue heterogeneity: how much repeated / similar content the user tolerates before
// fatigue bites. Higher => slower topic/format fatigue.
inline constexpr double kRepetitionToleranceLo = 0.0;
inline constexpr double kRepetitionToleranceHi = 1.0;
// P16-17 fatigue heterogeneity: appetite for genuinely new content before boredom sets in.
inline constexpr double kNoveltyToleranceLo = 0.0;
inline constexpr double kNoveltyToleranceHi = 1.0;
// P16-17 creator-fatigue immunity for favoured creators (loyal users tire of a creator slowly).
inline constexpr double kCreatorLoyaltyLo = 0.0;
inline constexpr double kCreatorLoyaltyHi = 1.0;
// P20 retention: strength of the daily-return habit (higher => stickier).
inline constexpr double kHabitStrengthLo = 0.0;
inline constexpr double kHabitStrengthHi = 1.0;
// P20 retention: baseline trust in the platform. Biased POSITIVE — most users start out trusting,
// so the starting range is [0.4, 1.0] rather than the full [0, 1] (trust is later eroded by
// regret).
inline constexpr double kPlatformTrustLo = 0.4;
inline constexpr double kPlatformTrustHi = 1.0;
// P20 retention: baseline usage intensity, in EXPECTED SESSIONS PER SIMULATED DAY. A light user
// opens the app ~0.5x/day, a heavy user ~6x/day.
inline constexpr double kBaselineDailyUsageLo = 0.5;
inline constexpr double kBaselineDailyUsageHi = 6.0;
// P20 exposure-driven preference evolution: how fast the hidden preference moves under exposure
// (higher => more malleable tastes).
inline constexpr double kPreferencePlasticityLo = 0.0;
inline constexpr double kPreferencePlasticityHi = 1.0;

} // namespace userTraitsV2

// Realism V2 user-side augmentation (V2 TDD 4.2 + 5, Phase 13). Called by generateDataset ONLY
// when realism.content_v2 is on, AFTER the V1 generators (and after the reel-side augmentation,
// whose ModalitySpaces it consumes as data — no draw on any reel stream). The V1 HiddenUserState
// fields were produced by the untouched V1 path, so gate-on leaves them byte-identical and
// gate-off performs zero V2 draws (D17). Consumes the caller-forked "users-v2" stream (D19);
// never calls forkRng. The public User struct is untouched this phase (no recommender-visible
// estimates yet).
//
// PACKAGE-B OWNERSHIP, FROZEN SIGNATURE: package B implements this in user_augmenter_v2.cpp
// (currently a scaffolding stub) and defines the rr::userTraitsV2 sampling-range constants as
// the single source of truth for every V2 trait's [lo, hi].
//
// Per user, in a fixed documented draw order: per-modality preference embeddings (L2-normalized
// blends of 1-3 centres from the SAME shared ModalitySpaces the reels use, plus noise), the
// scalar preference/susceptibility traits, primaryLanguage from
// rr::languageWeights(realism.languages), languageMismatchTolerance, and the forward traits
// (P16-17 tolerances, P20 retention/plasticity), each within its documented range.
void augmentUsersV2(std::vector<HiddenUserState> &hiddenStates, const ModalitySpaces &spaces,
                    const SimulationConfig &config, const RealismConfig &realism, Rng &usersV2Rng);

} // namespace rr
