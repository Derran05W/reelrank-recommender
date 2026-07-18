#pragma once

#include "rr/domain/interaction.hpp"

namespace rr {

// Everything the behaviour model decided about one shown reel (TDD 10). One impression can carry
// several signals at once (a completed watch that was also liked and shared); the boolean flags
// are the ground truth, and `primaryType` collapses them into the single InteractionType of the
// assembled event using this fixed priority (most significant first):
//   NotInterested > InstantSkip > Share > FollowCreator > Like > Rewatch > CompleteWatch >
//   PartialWatch > Impression.
//
// Lives in domain/ (not simulation/) because it is the OBSERVABLE outcome record shared across
// the simulation/learning boundary: RewardModel (learning/) consumes it without ever reaching
// simulator-owned hidden state — a reach the D18 include-graph guard would reject. baseAffinity
// and behaviourScore are derived from hidden state by the simulator; they ride on the outcome
// for the evaluation carve-out and tests, and no recommender-side consumer reads them.
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

} // namespace rr
