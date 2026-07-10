#pragma once

#include <cstdint>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// TDD 8.2.
struct Reel {
    ReelId id;
    CreatorId creatorId;

    Embedding embedding;

    float intrinsicQuality;
    float freshnessScore;
    float durationSeconds;

    TopicId primaryTopic;
    std::vector<TopicId> secondaryTopics;

    Timestamp createdAt;

    uint64_t impressionCount;
    uint64_t completionCount;
    uint64_t likeCount;
    uint64_t shareCount;
    uint64_t skipCount;

    bool active;
};

} // namespace rr
