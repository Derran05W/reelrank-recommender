#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
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
    // Per-feature ranking contributions (TDD 14.4), copied off the winning Candidate so per-item
    // explanations reach apps/tests. Appended LAST with a default member initializer so the
    // pre-Phase-6 positional aggregate initializers (RankedReel{id, score, rank, sources}) stay
    // valid AND warning-clean under -Wmissing-field-initializers (the default initializer exempts
    // an omitted trailing member from that warning; RankedReel stays an aggregate in C++20). Empty
    // on the nullptr-ranker (identity) orchestrator path; populated on the ranked path.
    std::unordered_map<std::string, float> featureContributions{};
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
