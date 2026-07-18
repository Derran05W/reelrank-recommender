#include "rr/simulation/user_augmenter_v2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/simulation/language.hpp"

namespace rr {

namespace {

// Sample one modality preference embedding: an L2-normalized blend of
// kMinModalityCentres..kMaxModalityCentres DISTINCT centres from `centres` (the SAME shared
// ModalitySpaces the reels blend), plus per-dimension gaussian noise. This mirrors the
// generateUsers topic-mix machinery (user_generator.cpp) exactly: draw a centre count, select that
// many distinct centres via partial Fisher-Yates, weight each by a base draw, accumulate, add
// gaussian noise, L2-normalize. Blending the SAME centres the reels use gives the modality channels
// learnable shared structure (without it, modality dot products would be structureless noise).
//
// FALLBACK (documented, deterministic): when `centres` is empty — a degenerate ModalitySpaces, or
// package A's Phase-13 stub before integration — the blend is skipped and the preference is a pure
// per-dimension gaussian unit vector. This is a REDUCED-DRAW path: the count / selection / weight
// draws (each of which requires >= 1 centre; uniformInt(0) would throw) are omitted, and only the
// `dim` gaussian noise draws and the normalization run. It stays deterministic and still yields a
// unit-norm dim-vector, so later-phase modality dot products remain well-formed.
Embedding sampleModalityPreference(const std::vector<Embedding> &centres, uint32_t dim, Rng &rng) {
    Embedding vec(dim, 0.0f);
    if (!centres.empty()) {
        const uint32_t numCentres = static_cast<uint32_t>(centres.size());
        const uint32_t span = static_cast<uint32_t>(userTraitsV2::kMaxModalityCentres -
                                                    userTraitsV2::kMinModalityCentres);
        const uint32_t desired = static_cast<uint32_t>(userTraitsV2::kMinModalityCentres) +
                                 static_cast<uint32_t>(rng.uniformInt(span + 1));
        const uint32_t m = std::min(desired, numCentres);

        // Select m distinct centres via partial Fisher-Yates over a fresh index permutation.
        std::vector<uint32_t> idx(numCentres);
        std::iota(idx.begin(), idx.end(), 0u);
        for (uint32_t k = 0; k < m; ++k) {
            const uint64_t j = k + rng.uniformInt(numCentres - k);
            std::swap(idx[k], idx[static_cast<std::size_t>(j)]);
        }

        // Weighted blend: weight = base in [kModalityWeightBaseLo, 1) per selected centre.
        for (uint32_t k = 0; k < m; ++k) {
            const double base = rng.uniform(userTraitsV2::kModalityWeightBaseLo, 1.0);
            const float a = static_cast<float>(base);
            const Embedding &centre = centres[idx[k]];
            for (uint32_t d = 0; d < dim; ++d) {
                vec[d] += a * centre[d];
            }
        }
    }

    // Additive per-dimension gaussian noise (always), then L2-normalize into the preference.
    for (uint32_t d = 0; d < dim; ++d) {
        vec[d] += static_cast<float>(rng.gaussian() * userTraitsV2::kModalityNoiseScale);
    }
    normalize(vec);
    return vec;
}

// One primary-language draw against the CUMULATIVE rr::languageWeights(languages) — the SAME skewed
// global distribution the reels sample, so the two sides' language mixes align by construction. A
// single uniform01 draw is bucketed through the cumulative weights; the last id is the catch-all
// for the floating-point tail so the draw never falls through. The draw COUNT is independent of
// `languages`, keeping the users-v2 draw sequence stable across language-set sizes.
LanguageId sampleLanguage(const std::vector<double> &cumulative, Rng &rng) {
    const double r = rng.uniform01();
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        if (r < cumulative[i]) {
            return LanguageId{static_cast<uint32_t>(i)};
        }
    }
    return LanguageId{static_cast<uint32_t>(cumulative.size() - 1)};
}

} // namespace

void augmentUsersV2(std::vector<HiddenUserState> &hiddenStates, const ModalitySpaces &spaces,
                    const SimulationConfig &config, const RealismConfig &realism, Rng &usersV2Rng) {
    const uint32_t dim = config.dimensions;

    // Cumulative language distribution, precomputed once (a pure function of realism.languages, no
    // rng): the per-user language draw buckets a single uniform01 through it. languageWeights
    // throws std::invalid_argument when realism.languages == 0 (a setup error, D10).
    std::vector<double> langCumulative = languageWeights(realism.languages);
    for (std::size_t i = 1; i < langCumulative.size(); ++i) {
        langCumulative[i] += langCumulative[i - 1];
    }

    // Per user, in user (index) order, a FIXED documented draw order on usersV2Rng; determinism
    // depends only on this order. The V1 HiddenUserState fields (userId, hiddenPreference,
    // preferredTopics, the seven V1 traits) are produced by the untouched V1 generateUsers path and
    // are NOT written here (D17: gate-on leaves every V1 field byte-identical). Only the V2
    // channels and traits below are set, and only from this "users-v2" stream (D19).
    for (HiddenUserState &h : hiddenStates) {
        // (a) Per-modality preference embeddings, pinned order visual -> music -> emotional, each
        //     blending the corresponding shared ModalitySpaces centres.
        h.visualPreference = sampleModalityPreference(spaces.visualCentres, dim, usersV2Rng);
        h.musicPreference = sampleModalityPreference(spaces.musicCentres, dim, usersV2Rng);
        h.emotionalPreference = sampleModalityPreference(spaces.emotionalCentres, dim, usersV2Rng);

        // (b) Scalar content-value preferences / susceptibility traits, each uniform in its range.
        h.usefulnessPreference = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kUsefulnessPreferenceLo, userTraitsV2::kUsefulnessPreferenceHi));
        h.humourPreference = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kHumourPreferenceLo, userTraitsV2::kHumourPreferenceHi));
        h.controversyTolerance = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kControversyToleranceLo, userTraitsV2::kControversyToleranceHi));
        h.noveltySeeking = static_cast<float>(
            usersV2Rng.uniform(userTraitsV2::kNoveltySeekingLo, userTraitsV2::kNoveltySeekingHi));
        h.clickbaitSusceptibility = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kClickbaitSusceptibilityLo, userTraitsV2::kClickbaitSusceptibilityHi));
        h.informationTolerance = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kInformationToleranceLo, userTraitsV2::kInformationToleranceHi));

        // (c) Primary language (one uniform01 draw bucketed through the cumulative weights) and the
        //     language mismatch tolerance.
        h.primaryLanguage = sampleLanguage(langCumulative, usersV2Rng);
        h.languageMismatchTolerance =
            static_cast<float>(usersV2Rng.uniform(userTraitsV2::kLanguageMismatchToleranceLo,
                                                  userTraitsV2::kLanguageMismatchToleranceHi));

        // (d) Forward traits (generated now, consumed by the named later phase).
        h.repetitionTolerance = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kRepetitionToleranceLo, userTraitsV2::kRepetitionToleranceHi));
        h.noveltyTolerance = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kNoveltyToleranceLo, userTraitsV2::kNoveltyToleranceHi));
        h.creatorLoyalty = static_cast<float>(
            usersV2Rng.uniform(userTraitsV2::kCreatorLoyaltyLo, userTraitsV2::kCreatorLoyaltyHi));
        h.habitStrength = static_cast<float>(
            usersV2Rng.uniform(userTraitsV2::kHabitStrengthLo, userTraitsV2::kHabitStrengthHi));
        h.platformTrust = static_cast<float>(
            usersV2Rng.uniform(userTraitsV2::kPlatformTrustLo, userTraitsV2::kPlatformTrustHi));
        h.baselineDailyUsage = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kBaselineDailyUsageLo, userTraitsV2::kBaselineDailyUsageHi));
        h.preferencePlasticity = static_cast<float>(usersV2Rng.uniform(
            userTraitsV2::kPreferencePlasticityLo, userTraitsV2::kPreferencePlasticityHi));
    }
}

} // namespace rr
