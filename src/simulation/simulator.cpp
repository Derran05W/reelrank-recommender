#include "rr/simulation/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace rr {

namespace {

// Fixed per-impression browsing overhead added to the logical clock on every step, on top of the
// (rounded) time spent watching the reel. Models the seconds a user spends scrolling to and
// deciding on the next reel, and guarantees the clock strictly advances even on a zero-watch
// instant skip (D9 — simulated seconds, never wall clock).
constexpr Timestamp kBrowseOverheadSeconds = 2;

// Spread of per-session target lengths around hidden.avgSessionLength, as a fraction of the mean.
// Session lengths vary run-to-run but stay centred on the user's trait.
constexpr double kSessionLengthRelStddev = 0.25;

// Exponential decay factor 2^(-dt/halfLife) (dt in simulated seconds, D9) that brings a reel's
// trending accumulators forward from `updatedAt` to `now`, clamping to 1 when now <= updatedAt
// (TDD 12.4). This is the WRITE side of the trending twin; it intentionally MIRRORS
// rr::trendingDecayFactor (scoring.hpp — the READ side used by rr::trendingScore) rather than
// calling it, so rr_simulation stays free of the rr_recommend/vector-db link dependency (the
// module boundary declared in CMakeLists; inspect_user links rr_simulation alone). The two are
// kept in lockstep by the shared half-life semantics and cross-checked in simulator_test against
// rr::trendingDecayFactor as an independent oracle.
double trendingDecayForward(Timestamp updatedAt, Timestamp now, double halfLifeSeconds) {
    const double dt = now > updatedAt ? static_cast<double>(now - updatedAt) : 0.0;
    return std::exp2(-dt / halfLifeSeconds);
}

// Per-impression gains applied to the recommender-visible User::creatorAffinity estimate (TDD
// 12.6) from the OBSERVABLE outcome flags only — never from HiddenUserState (D11: the ground-truth
// creator affinity C_{u,c} is off-limits here; this is the system's LOGGED estimate). Signals are
// weighted by how strongly they express enjoyment of the creator: an explicit follow most, a plain
// completed watch least, an explicit "not interested" pushes the estimate down. The running
// per-creator affinity is accumulated and then clamped to [0, 1]; that CONTRACT is relied on by
// the sibling ranker package (the creator-affinity ranking feature) and by
// CreatorAffinityCandidateSource, so it must hold on every write.
constexpr float kAffinityFollowedGain = 0.25f;
constexpr float kAffinitySharedGain = 0.15f;
constexpr float kAffinityLikedGain = 0.10f;
constexpr float kAffinityCompletedGain = 0.02f;
constexpr float kAffinityNotInterestedGain = -0.20f;

} // namespace

Simulator::Simulator(const BehaviourConfig &behaviour, const RewardConfig &reward, Rng rng,
                     uint32_t recentWindow, double trendingHalfLifeSeconds)
    : behaviour_(behaviour), reward_(reward), rng_(std::move(rng)), recentWindow_(recentWindow),
      trendingHalfLifeSeconds_(trendingHalfLifeSeconds) {}

uint32_t Simulator::sampleSessionTarget(float avgSessionLength) {
    // Gaussian around the mean with a proportional spread, clamped to at least one interaction so a
    // session always contains the reel that opened it. Uses rng_ so the sequence is deterministic.
    const double mean = std::max(1.0, static_cast<double>(avgSessionLength));
    const double sampled = mean + rng_.gaussian() * (mean * kSessionLengthRelStddev);
    const long rounded = std::lround(sampled);
    return static_cast<uint32_t>(std::max<long>(1, rounded));
}

StepResult Simulator::step(User &user, const HiddenUserState &hidden, Reel &reel,
                           const Creator &creator) {
    StepResult result{};

    // 1. Ground-truth reaction. HiddenUserState is handed only to the behaviour model here (D11).
    result.outcome = behaviour_.simulate(hidden, reel, creator, rng_);
    const BehaviourOutcome &outcome = result.outcome;

    // 2. Reward: pure function of the outcome, no extra randomness.
    const float reward = reward_.reward(outcome);

    // 3. Advance the logical clock by the watched seconds (rounded) plus browse overhead (D9). The
    //    clock therefore advances by at least kBrowseOverheadSeconds every step.
    const Timestamp watchedSeconds =
        static_cast<Timestamp>(std::lround(std::max(0.0f, outcome.watchSeconds)));
    now_ += watchedSeconds + kBrowseOverheadSeconds;

    // 4. Resolve the session this interaction belongs to. On first sight of the user we open a
    //    session; when the running currentSessionLength has reached the target we rotate to a new
    //    session (advance the id, reset the length, resample the target) BEFORE attributing this
    //    interaction, so this reel opens the new session.
    auto [it, inserted] = sessions_.try_emplace(user.id);
    SessionState &session = it->second;
    if (inserted) {
        session.sessionId = SessionId{0};
        session.targetLength = sampleSessionTarget(hidden.avgSessionLength);
        user.currentSessionLength = 0;
    } else if (user.currentSessionLength >= session.targetLength) {
        session.sessionId = SessionId{session.sessionId.value + 1};
        session.targetLength = sampleSessionTarget(hidden.avgSessionLength);
        user.currentSessionLength = 0;
    }

    // 5. Assemble the recommender-visible event. Every field is an observable interaction signal;
    //    no hidden preference vector or trait leaks through (D11).
    result.event = InteractionEvent{user.id,
                                    reel.id,
                                    reel.creatorId,
                                    outcome.primaryType,
                                    outcome.watchSeconds,
                                    outcome.watchRatio,
                                    reward,
                                    now_,
                                    session.sessionId};

    // 6. Update reel ground-truth counters, consistent with the outcome flags.
    reel.impressionCount += 1;
    if (outcome.completed) {
        reel.completionCount += 1;
    }
    if (outcome.liked) {
        reel.likeCount += 1;
    }
    if (outcome.shared) {
        reel.shareCount += 1;
    }
    if (outcome.instantSkip) {
        reel.skipCount += 1;
    }

    // 6b. Maintain the reel's trending accumulators (TDD 12.4): the exponentially-decayed twin of
    //     the popularity numerator. Decay both accumulators forward to this event's timestamp,
    //     then add exactly the increment the lifetime popularity counters gained for this event
    //     (+1 iff completed, +2 iff liked, +4 iff shared — the same 1/2/4 weights as
    //     popularityEngagement). rr::trendingScore reads these decayed forward again at query time,
    //     so the velocity is correct at any later `now` without a per-query update. Pure
    //     arithmetic on existing state — no rng draw (D8).
    const double decay =
        trendingDecayForward(reel.trendingUpdatedAt, now_, trendingHalfLifeSeconds_);
    reel.trendingEngagement *= decay;
    reel.trendingImpressions *= decay;
    reel.trendingImpressions += 1.0;
    reel.trendingEngagement += (outcome.completed ? 1.0 : 0.0) + (outcome.liked ? 2.0 : 0.0) +
                               (outcome.shared ? 4.0 : 0.0);
    reel.trendingUpdatedAt = now_; // set last, so the decay above used the PREVIOUS update time

    // 7. Update recommender-visible user bookkeeping.
    user.seenReels.insert(reel.id);
    user.totalInteractions += 1;
    user.currentSessionLength += 1;
    user.recentInteractions.push_back(result.event);
    while (user.recentInteractions.size() > recentWindow_) {
        user.recentInteractions.pop_front();
    }

    // 7b. Update the recommender-visible creator-affinity estimate from the OBSERVABLE outcome
    //     flags only (never HiddenUserState — D11). Accumulate the signed gains, then clamp the
    //     running per-creator estimate to [0, 1] (contract for the ranker + creator source). The
    //     map is only touched when some signal fired, so plain skips leave it untouched.
    const float affinityDelta = (outcome.followed ? kAffinityFollowedGain : 0.0f) +
                                (outcome.shared ? kAffinitySharedGain : 0.0f) +
                                (outcome.liked ? kAffinityLikedGain : 0.0f) +
                                (outcome.completed ? kAffinityCompletedGain : 0.0f) +
                                (outcome.notInterested ? kAffinityNotInterestedGain : 0.0f);
    if (affinityDelta != 0.0f) {
        float &affinity = user.creatorAffinity[reel.creatorId];
        affinity = std::clamp(affinity + affinityDelta, 0.0f, 1.0f);
    }

    return result;
}

Timestamp Simulator::now() const { return now_; }

} // namespace rr
