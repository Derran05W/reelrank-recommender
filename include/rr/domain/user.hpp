#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/interaction.hpp"

namespace rr {

// TDD 8.3, minus hiddenPreference (design decision D11): the recommender-visible User carries no
// hidden state. The simulator's ground-truth preference lives in HiddenUserState instead.
struct User {
    UserId id;

    Embedding estimatedPreference;
    Embedding longTermPreference;
    Embedding sessionPreference;

    std::unordered_set<ReelId> seenReels;
    std::unordered_map<CreatorId, float> creatorAffinity;

    std::deque<InteractionEvent> recentInteractions;

    uint64_t totalInteractions;
    uint64_t currentSessionLength;
};

} // namespace rr
