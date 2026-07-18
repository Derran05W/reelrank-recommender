#pragma once

#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/user.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"

namespace rr {

// TDD 11.1 cold-start prior: the normalized mean of the population's ground-truth preferences.
// This is evaluation/setup-side hidden-state access (TDD 18.2's carve-out): the aggregate stands
// in for the platform-wide average *learned* preference a real system would observe; individual
// hidden vectors never reach a recommender (D11). Throws std::invalid_argument if `hidden` is
// empty or the mean has no direction.
Embedding globalAveragePreference(const std::vector<HiddenUserState> &hidden);

// Apply the cold-start prior to every user: estimatedPreference, longTermPreference, and
// sessionPreference all start at `prior`. From Phase 7 the OnlineUserStateUpdater evolves all
// three per interaction (when LearningConfig.enabled); with learning disabled the estimates stay
// frozen at this prior for the whole run — the pre-Phase-7 baseline behaviour.
void applyColdStart(std::vector<User> &users, const Embedding &prior);

} // namespace rr
