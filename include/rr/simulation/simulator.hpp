#pragma once

#include <unordered_map>

#include "rr/domain/creator.hpp"
#include "rr/domain/hidden_user_state.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/learning/reward_model.hpp"
#include "rr/simulation/behaviour_model.hpp"

namespace rr {

// One simulated impression: the ground-truth outcome plus the assembled event. The outcome is
// simulator/evaluation-side only (it derives from hidden state, TDD 18.2); recommenders may see
// the event but never the outcome.
struct StepResult {
    BehaviourOutcome outcome;
    InteractionEvent event;
};

// Drives the ground-truth interaction loop (TDD 10 + phase-3 task 5): given a user shown a reel,
// it simulates the reaction, computes the reward, assembles the InteractionEvent, advances the
// logical clock (D9 — simulated seconds, never wall clock), and updates reel counters
// (impressions/completions/likes/shares/skips) and recommender-visible user bookkeeping
// (seenReels, recentInteractions, totalInteractions, currentSessionLength, session rotation
// around hidden.avgSessionLength).
class Simulator {
  public:
    // `rng` is forked by the caller on the "behaviour" stream (D8); the Simulator owns it from
    // here on. `recentWindow` bounds User::recentInteractions (LearningConfig.recentWindow).
    Simulator(const BehaviourConfig &behaviour, const RewardConfig &reward, Rng rng,
              uint32_t recentWindow);

    // Simulate `user` being shown `reel`. Mutates `reel` counters and `user` bookkeeping;
    // advances the logical clock. HiddenUserState is read-only here and only ever handed to the
    // BehaviourModel (D11).
    StepResult step(User &user, const HiddenUserState &hidden, Reel &reel, const Creator &creator);

    // Current logical time in simulated seconds (starts at 0).
    Timestamp now() const;

  private:
    // Per-user session bookkeeping. The recommender-visible User carries currentSessionLength but
    // not the SessionId, so the Simulator owns the id and the (rng-sampled) target length that
    // triggers rotation. avgSessionLength (hidden state) is read here ONLY to derive targetLength;
    // it is never copied into any recommender-visible field (D11).
    struct SessionState {
        SessionId sessionId{0};
        uint32_t targetLength = 1;
    };

    // Draw a per-session target length from hidden.avgSessionLength using rng_ (deterministic, D8).
    uint32_t sampleSessionTarget(float avgSessionLength);

    BehaviourModel behaviour_;
    RewardModel reward_;
    Rng rng_;
    uint32_t recentWindow_;
    Timestamp now_ = 0;
    std::unordered_map<UserId, SessionState> sessions_;
};

} // namespace rr
