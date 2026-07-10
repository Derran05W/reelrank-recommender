#pragma once

#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// TDD 8.4.
enum class InteractionType {
    Impression,
    InstantSkip,
    PartialWatch,
    CompleteWatch,
    Rewatch,
    Like,
    Share,
    FollowCreator,
    NotInterested
};

struct InteractionEvent {
    UserId userId;
    ReelId reelId;
    CreatorId creatorId;

    InteractionType type;

    float watchSeconds;
    float watchRatio;
    float reward;

    Timestamp timestamp;
    SessionId sessionId;
};

} // namespace rr
