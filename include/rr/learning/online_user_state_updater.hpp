#pragma once

#include <vector>

#include "rr/domain/reel.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/learning/user_state_updater.hpp"

namespace rr {

// Online preference learning (TDD 11.2, 11.3, 8.3): the Phase 7 UserStateUpdater. Stateless and
// deterministic (D8: no rng; D9: no clock) - every apply() is a pure function of the User's
// recorded interaction history and the reel catalog.
//
// Call-site contract: apply() runs AFTER Simulator::step has recorded the interaction, so
// `interaction` is already the newest entry of user.recentInteractions and seenReels /
// creatorAffinity / interaction counters are Simulator-maintained (Phase 3/6 ownership, kept -
// recorded as a Phase 7 deviation from the TDD 23.4 wording). This class owns ONLY the three
// preference vectors:
//
//  1. longTermPreference <- normalize((1 - longTermRate) * u + longTermRate * reward * v)
//     (TDD 11.2; negative reward pushes away).
//  2. sessionPreference  <- normalize(sum over the recent-interaction window's CURRENT-session
//     events i of sessionLambda^(n-i) * reward_i * embedding_i) (TDD 11.3). Restricting the sum
//     to events with this interaction's sessionId makes the between-session reset (phase task 3)
//     implicit: rotation starts the sum over.
//  3. estimatedPreference <- normalize(longTermWeight * longTerm + sessionWeight * session)
//     (TDD 8.3). estimatedPreference is thus the CACHED effective preference, so
//     rr::effectivePreference(user) keeps returning it by const reference.
//
// All three vectors remain unit-length after every apply (property-tested). Degenerate
// normalizations (near-zero direction) fall back deterministically; the rules are documented at
// their definitions in online_user_state_updater.cpp.
//
// `reels` is the dense-id catalog (reels[i].id.value == i, the dataset-generator guarantee the
// whole pipeline relies on) used to look up embeddings of past in-window interactions.
class OnlineUserStateUpdater final : public UserStateUpdater {
  public:
    OnlineUserStateUpdater(const std::vector<Reel> &reels, const LearningConfig &config);

    void apply(User &user, const Reel &reel, const InteractionEvent &interaction) const override;

  private:
    const std::vector<Reel> &reels_;
    LearningConfig config_;
};

} // namespace rr
