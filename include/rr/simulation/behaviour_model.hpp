#pragma once

#include "rr/domain/behaviour_outcome.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/reel.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"

namespace rr {

// BehaviourOutcome moved to rr/domain/behaviour_outcome.hpp (Phase 13): it is the observable
// outcome record shared with learning/, which must not reach this hidden-state-including header
// (D18 include-graph guard).

// The synthetic ground-truth user (TDD 10). This is the ONLY component that reads
// HiddenUserState (design decision D11) — it never sees estimated preferences.
//
// Ground-truth creator affinity C_{u,c} = dot(hidden.hiddenPreference, creator.styleEmbedding);
// the recommender-side User::creatorAffinity estimate is never consulted here. The duration
// penalty D_v is in [0, 1], increasing in reel.durationSeconds and decreasing in
// hidden.durationTolerance (exact shape is an implementation constant, unit-tested).
class BehaviourModel {
  public:
    explicit BehaviourModel(const BehaviourConfig &config);

    // Stochastic-but-reproducible reaction to one shown reel: identical (hidden, reel, creator,
    // rng state) produces an identical outcome (TDD 24.6). The caller owns rng stream forking
    // (D8) — this class never calls forkRng.
    BehaviourOutcome simulate(const HiddenUserState &hidden, const Reel &reel,
                              const Creator &creator, Rng &rng) const;

  private:
    BehaviourConfig config_;
};

} // namespace rr
