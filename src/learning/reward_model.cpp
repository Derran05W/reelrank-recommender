#include "rr/learning/reward_model.hpp"

#include <algorithm>
#include <cmath>

namespace rr {

namespace {

// Normalization anchor for the log-watch-seconds term (TDD 10.5: "bounded or normalized to avoid
// unstable preference updates"). The Phase 2 reel generator draws durations from four buckets whose
// upper edge is 120 s (src/simulation/reel_generator.cpp kDurationBuckets = {..,{60,120}}), so 120
// s is the maximum single-view watch time. We normalize log1p(watchSeconds) by log1p(120) so a
// full-length watch contributes ~1.0 to the term (before its 0.15 weight) and can never dominate
// the reward. Rewatches (watchRatio > 1) push watchSeconds slightly past 120 s, nudging the ratio
// just above 1.0 — harmless because the final reward is clamped to [-1, 1] regardless.
constexpr double kMaxReelDurationSeconds = 120.0;
const double kLogWatchNorm = std::log1p(kMaxReelDurationSeconds);

} // namespace

RewardModel::RewardModel(const RewardConfig &config) : config_(config) {}

float RewardModel::reward(const BehaviourOutcome &outcome) const {
    // Pure, deterministic function of the outcome's engagement signals (TDD 10.5). Accumulated in
    // double, then clamped to [-1, 1] so downstream preference updates (Phase 7) stay stable.
    const double watchRatioClamped = std::clamp(static_cast<double>(outcome.watchRatio), 0.0, 1.0);
    const double normalizedLogWatch =
        std::log1p(std::max(0.0, static_cast<double>(outcome.watchSeconds))) / kLogWatchNorm;

    double r = 0.0;
    r += config_.watchRatioWeight * watchRatioClamped;
    r += config_.watchSecondsWeight * normalizedLogWatch;
    r += config_.likeWeight * (outcome.liked ? 1.0 : 0.0);
    r += config_.shareWeight * (outcome.shared ? 1.0 : 0.0);
    r += config_.followWeight * (outcome.followed ? 1.0 : 0.0);
    r -= config_.instantSkipPenalty * (outcome.instantSkip ? 1.0 : 0.0);
    r -= config_.notInterestedPenalty * (outcome.notInterested ? 1.0 : 0.0);

    return std::clamp(static_cast<float>(r), -1.0f, 1.0f);
}

} // namespace rr
