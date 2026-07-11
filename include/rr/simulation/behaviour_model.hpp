#pragma once

#include "rr/domain/creator.hpp"
#include "rr/domain/hidden_user_state.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"

namespace rr {

// Everything the behaviour model decided about one shown reel (TDD 10). One impression can carry
// several signals at once (a completed watch that was also liked and shared); the boolean flags
// are the ground truth, and `primaryType` collapses them into the single InteractionType of the
// assembled event using this fixed priority (most significant first):
//   NotInterested > InstantSkip > Share > FollowCreator > Like > Rewatch > CompleteWatch >
//   PartialWatch > Impression.
struct BehaviourOutcome {
    float baseAffinity;   // a = p_u . q_v (TDD 10.1) — hidden preference vs reel embedding
    float behaviourScore; // z = alpha*a + beta*Q + gamma*C - delta*D + eps (TDD 10.2)

    bool instantSkip;   // fired P(instantSkip) = sigmoid(-z + skipBias)     (TDD 10.3)
    bool completed;     // fired P(complete) = sigmoid(z)                    (TDD 10.3)
    bool rewatch;       // watchRatio > 1.0                                  (TDD 10.4)
    bool liked;         // requires completed watch; propensity-modulated    (TDD 10.3)
    bool shared;        // requires completed watch; propensity-modulated    (TDD 10.3)
    bool followed;      // requires completed watch + high creator affinity
    bool notInterested; // low-probability path, only for very negative z

    float watchRatio;   // fraction of durationSeconds watched; > 1.0 means rewatch (TDD 10.4)
    float watchSeconds; // watchRatio * reel.durationSeconds

    InteractionType primaryType; // collapsed per the priority above
};

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
