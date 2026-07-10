#pragma once

#include <cstddef>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// TDD 8.6.
struct RecommendationRequest {
    UserId userId;
    SessionId sessionId;

    size_t feedSize;
    size_t candidateLimit;

    bool enableExploration;
    bool enableDiversity;

    Timestamp requestTime;
};

// TDD 8.7.
struct RankedReel {
    ReelId reelId;
    float score;
    size_t rank;
    std::vector<CandidateSource> sources;
};

struct RecommendationResponse {
    std::vector<RankedReel> reels;

    double retrievalLatencyMs;
    double rankingLatencyMs;
    double rerankingLatencyMs;
    double totalLatencyMs;

    size_t candidatesRetrieved;
    size_t candidatesRanked;
};

} // namespace rr
