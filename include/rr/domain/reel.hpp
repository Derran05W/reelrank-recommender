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

    // Trending accumulator (TDD 12.4): exponentially decayed interaction counters maintained by
    // Simulator::step and read via rr::trendingScore (scoring.hpp). On every impression at time t
    // both accumulators are decayed by trendingDecayFactor(trendingUpdatedAt, t, halfLife), then
    // trendingImpressions += 1 and trendingEngagement += the event's engagement increment (same
    // 1/2/4 completion/like/share weights as popularityEngagement). Appended after `active` with
    // defaults so pre-Phase-6 positional aggregate initializers stay valid.
    double trendingEngagement = 0.0;
    double trendingImpressions = 0.0;
    Timestamp trendingUpdatedAt = 0;
};

} // namespace rr
