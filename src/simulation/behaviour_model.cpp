#include "rr/simulation/behaviour_model.hpp"

#include <algorithm>
#include <cmath>

#include "rr/core/embedding.hpp"

namespace rr {
namespace {

// --- Duration penalty D_v (TDD 10.2, header contract) -----------------------------------------
// D_v in [0, 1] increases with reel.durationSeconds and decreases with hidden.durationTolerance.
// Shape: normalize the duration to [0, 1] over the generator's [5, 120)s span (reel_generator.cpp
// kDurationBuckets), then let a fully duration-tolerant user shed a fixed fraction of that penalty:
//     dNorm = clamp((duration - kMinDurationSeconds) / (kMaxDurationSeconds - kMinDurationSeconds))
//     D_v   = dNorm * (1 - kDurationToleranceRelief * tolerance)
// This is monotonically increasing in duration and decreasing in tolerance, and stays within
// [0, dNorm] subset of [0, 1] because tolerance in [0, 1] and kDurationToleranceRelief in [0, 1].
constexpr double kMinDurationSeconds = 5.0;
constexpr double kMaxDurationSeconds = 120.0;
constexpr double kDurationToleranceRelief = 0.80;

// --- Watch-ratio bands (TDD 10.4), sampled uniformly within the selected [lo, hi) band ---------
// instantSkip forces a tiny ratio; completed maps to the high band (so completed <=> high band and
// rewatch (>1.0) only ever occurs on a completed watch); every other (partial) impression draws a
// band selected by base affinity: very-low / low / medium.
constexpr double kInstantSkipRatioLo = 0.00;
constexpr double kInstantSkipRatioHi = 0.05;
constexpr double kVeryLowRatioLo = 0.00;
constexpr double kVeryLowRatioHi = 0.10;
constexpr double kLowRatioLo = 0.10;
constexpr double kLowRatioHi = 0.40;
constexpr double kMediumRatioLo = 0.40;
constexpr double kMediumRatioHi = 0.80;
constexpr double kHighRatioLo = 0.80;
constexpr double kHighRatioHi = 1.20;

// Base-affinity thresholds that pick the partial-watch band. Below Low => very-low band; below
// Medium => low band; otherwise medium band.
constexpr float kPartialBandLowAffinity = 0.05f;
constexpr float kPartialBandMediumAffinity = 0.25f;

// A non-skip impression whose sampled watch ratio is below this collapses to Impression; at or
// above it collapses to (at least) PartialWatch.
constexpr double kImpressionMaxRatio = 0.02;

// --- Engagement modulation (TDD 10.3) ---------------------------------------------------------
// like/share/follow require a completed watch. like/share scale their per-user propensity by an
// affinity boost in [1, 1 + gain]; follow needs high true creator affinity C and scales a base
// probability by C.
constexpr double kEngagementAffinityGain = 2.0;
constexpr double kFollowBaseProb = 0.10;
constexpr double kFollowAffinityGain = 1.0;
constexpr float kFollowCreatorAffinityThreshold = 0.30f;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// D_v in [0, 1], increasing in duration, decreasing in tolerance.
double durationPenalty(float durationSeconds, float durationTolerance) {
    const double span = kMaxDurationSeconds - kMinDurationSeconds;
    const double dNorm =
        clamp01((static_cast<double>(durationSeconds) - kMinDurationSeconds) / span);
    const double relief = kDurationToleranceRelief * clamp01(durationTolerance);
    return dNorm * (1.0 - relief);
}

struct Band {
    double lo;
    double hi;
};

// Select the watch-ratio band from the outcome category and base affinity.
Band selectBand(bool instantSkip, bool completed, float baseAffinity) {
    if (instantSkip) {
        return {kInstantSkipRatioLo, kInstantSkipRatioHi};
    }
    if (completed) {
        return {kHighRatioLo, kHighRatioHi};
    }
    if (baseAffinity < kPartialBandLowAffinity) {
        return {kVeryLowRatioLo, kVeryLowRatioHi};
    }
    if (baseAffinity < kPartialBandMediumAffinity) {
        return {kLowRatioLo, kLowRatioHi};
    }
    return {kMediumRatioLo, kMediumRatioHi};
}

// Collapse the ground-truth flags into a single InteractionType using the header's fixed priority:
//   NotInterested > InstantSkip > Share > FollowCreator > Like > Rewatch > CompleteWatch >
//   PartialWatch > Impression.
InteractionType collapsePrimaryType(const BehaviourOutcome &o) {
    if (o.notInterested) {
        return InteractionType::NotInterested;
    }
    if (o.instantSkip) {
        return InteractionType::InstantSkip;
    }
    if (o.shared) {
        return InteractionType::Share;
    }
    if (o.followed) {
        return InteractionType::FollowCreator;
    }
    if (o.liked) {
        return InteractionType::Like;
    }
    if (o.rewatch) {
        return InteractionType::Rewatch;
    }
    if (o.completed) {
        return InteractionType::CompleteWatch;
    }
    if (o.watchRatio >= static_cast<float>(kImpressionMaxRatio)) {
        return InteractionType::PartialWatch;
    }
    return InteractionType::Impression;
}

} // namespace

BehaviourModel::BehaviourModel(const BehaviourConfig &config) : config_(config) {}

// Draw composition (fixed, documented sequence; deterministic given inputs + rng state, D8/TDD
// 24.6). Every rng draw below is taken in exactly this order:
//   1. gaussian()  -> noise eps, forming z.
//   2. bernoulli() -> notInterested, but only when z < notInterestedZ (a z-thresholded draw).
//   3. bernoulli() -> instantSkip = sigmoid(-z + skipBias).
//   4. bernoulli() -> completion draw = sigmoid(z); completed = draw && !instantSkip.
//   5. uniform01() -> position within the selected watch-ratio band.
//   6. (only when completed) bernoulli() x3 -> like, share, follow draws.
BehaviourOutcome BehaviourModel::simulate(const HiddenUserState &hidden, const Reel &reel,
                                          const Creator &creator, Rng &rng) const {
    BehaviourOutcome o{};

    // Base affinity a = p_u . q_v (TDD 10.1) and true creator affinity C = p_u . styleEmbedding.
    o.baseAffinity = dot(hidden.hiddenPreference, reel.embedding);
    const float creatorAffinity = dot(hidden.hiddenPreference, creator.styleEmbedding);

    // Behaviour score z = alpha*a + beta*Q + gamma*C - delta*D + noiseStd*eps (TDD 10.2).
    const double duration = durationPenalty(reel.durationSeconds, hidden.durationTolerance);
    const double eps = rng.gaussian();
    const double z = config_.alpha * o.baseAffinity + config_.beta * reel.intrinsicQuality +
                     config_.gamma * creatorAffinity - config_.delta * duration +
                     config_.noiseStd * eps;
    o.behaviourScore = static_cast<float>(z);

    // NotInterested: rare path, only for very negative z (TDD 10.3 low-probability path).
    o.notInterested =
        (z < config_.notInterestedZ) ? rng.bernoulli(config_.notInterestedProb) : false;

    // Instant skip and completion (TDD 10.3). Draw both so the rng sequence is branch-free here;
    // enforce mutual exclusion (instantSkip => !completed) via the flag combination.
    o.instantSkip = rng.bernoulli(sigmoid(-z + config_.skipBias));
    const bool completionDraw = rng.bernoulli(sigmoid(z));
    o.completed = completionDraw && !o.instantSkip;

    // Watch ratio: uniform draw inside the affinity/category-selected band (TDD 10.4).
    const Band band = selectBand(o.instantSkip, o.completed, o.baseAffinity);
    const double bandPos = rng.uniform01();
    o.watchRatio = static_cast<float>(band.lo + bandPos * (band.hi - band.lo));
    o.watchSeconds = o.watchRatio * reel.durationSeconds;
    o.rewatch = o.watchRatio > 1.0f;

    // Engagement (TDD 10.3): only on a completed watch, modulated by affinity + per-user traits.
    if (o.completed) {
        const double boost = 1.0 + kEngagementAffinityGain * clamp01(o.baseAffinity);
        o.liked = rng.bernoulli(clamp01(hidden.likePropensity * boost));
        o.shared = rng.bernoulli(clamp01(hidden.sharePropensity * boost));
        const double followProb =
            clamp01(kFollowBaseProb * (1.0 + kFollowAffinityGain * clamp01(creatorAffinity)));
        const bool followDraw = rng.bernoulli(followProb);
        o.followed = followDraw && (creatorAffinity > kFollowCreatorAffinityThreshold);
    }

    o.primaryType = collapsePrimaryType(o);
    return o;
}

} // namespace rr
