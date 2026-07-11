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

} // namespace

Simulator::Simulator(const BehaviourConfig &behaviour, const RewardConfig &reward, Rng rng,
                     uint32_t recentWindow)
    : behaviour_(behaviour), reward_(reward), rng_(std::move(rng)), recentWindow_(recentWindow) {}

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

    // 7. Update recommender-visible user bookkeeping.
    user.seenReels.insert(reel.id);
    user.totalInteractions += 1;
    user.currentSessionLength += 1;
    user.recentInteractions.push_back(result.event);
    while (user.recentInteractions.size() > recentWindow_) {
        user.recentInteractions.pop_front();
    }

    return result;
}

Timestamp Simulator::now() const { return now_; }

} // namespace rr
