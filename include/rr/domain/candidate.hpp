#pragma once

#include <string>
#include <unordered_map>

#include "rr/domain/ids.hpp"

namespace rr {

// TDD 8.5.
enum class CandidateSource {
    VectorHNSW,
    VectorExact,
    Popular,
    Trending,
    Fresh,
    CreatorAffinity,
    Exploration
};

struct Candidate {
    ReelId reelId;

    CandidateSource source;

    float retrievalDistance;
    float retrievalSimilarity;

    float rankingScore;

    std::unordered_map<std::string, float> featureContributions;
};

} // namespace rr
