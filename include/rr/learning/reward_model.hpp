#pragma once

#include "rr/domain/behaviour_outcome.hpp"
#include "rr/infrastructure/config.hpp"

namespace rr {

// TDD 10.5 reward: a weighted sum of the outcome's engagement signals, clamped to [-1, 1] so
// preference updates (Phase 7) stay stable. Weights come from RewardConfig; the log-watch-seconds
// term's normalization is documented at the implementation.
class RewardModel {
  public:
    explicit RewardModel(const RewardConfig &config);

    // Pure function of the outcome — no randomness, no hidden-state access.
    float reward(const BehaviourOutcome &outcome) const;

  private:
    RewardConfig config_;
};

} // namespace rr
