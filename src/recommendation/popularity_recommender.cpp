#include "rr/recommendation/popularity_recommender.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/recommendation/scoring.hpp"
#include "rr/recommendation/seen_filter.hpp"

namespace rr {

PopularityRecommender::PopularityRecommender(const RecommenderDeps &deps, Rng /*rng*/)
    : reels_(deps.reels), users_(deps.users) {}

double popularityEngagement(const Reel &reel) {
    return static_cast<double>(reel.completionCount) + 2.0 * static_cast<double>(reel.likeCount) +
           4.0 * static_cast<double>(reel.shareCount);
}

double smoothedPopularity(const Reel &reel, double priorMean, double pseudoImpressions) {
    const double engagement = popularityEngagement(reel);
    return (engagement + pseudoImpressions * priorMean) /
           (1.0 + static_cast<double>(reel.impressionCount) + pseudoImpressions);
}

RecommendationResponse PopularityRecommender::recommend(const RecommendationRequest &request) {
    const User &user = users_[request.userId.value];

    // Prior mean m = global engagement / global impressions over the current live counters,
    // falling back to 0 before any impression exists (TDD 12.3 Bayesian smoothing).
    const double priorMean = engagementPriorMean(reels_);

    // Score every eligible reel; rank by descending score, ties by ascending ReelId (so an
    // all-zero cold-start feed is the first feedSize reels in id order - fully deterministic).
    struct Scored {
        double score;
        ReelId id;
    };
    const std::vector<std::size_t> pool = eligibleReelIndices(reels_, user);
    std::vector<Scored> scored;
    scored.reserve(pool.size());
    for (std::size_t idx : pool) {
        scored.push_back(Scored{smoothedPopularity(reels_[idx], priorMean), reels_[idx].id});
    }
    std::sort(scored.begin(), scored.end(), [](const Scored &a, const Scored &b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.id.value < b.id.value;
    });

    RecommendationResponse response{};
    response.candidatesRetrieved = scored.size();
    response.candidatesRanked = scored.size();
    const std::size_t k = std::min(static_cast<std::size_t>(request.feedSize), scored.size());
    response.reels.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        response.reels.push_back(RankedReel{
            scored[i].id, static_cast<float>(scored[i].score), i, {CandidateSource::Popular}});
    }
    return response;
}

std::string PopularityRecommender::name() const { return "popularity"; }

} // namespace rr
