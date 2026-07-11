#include "rr/recommendation/scoring.hpp"

#include <cmath>

#include "rr/recommendation/popularity_recommender.hpp"

namespace rr {

namespace {

// 2^(-dt/halfLife) for dt >= 0. std::exp2 keeps the half-life semantics exact: dt == halfLife
// halves the value. Underflows cleanly to 0 for enormous dt (never NaN/Inf for finite inputs).
double halfLifeDecay(double dtSeconds, double halfLifeSeconds) {
    return std::exp2(-dtSeconds / halfLifeSeconds);
}

} // namespace

double freshnessScore(Timestamp createdAt, Timestamp now, double halfLifeSeconds) {
    const double age = now > createdAt ? static_cast<double>(now - createdAt) : 0.0;
    return halfLifeDecay(age, halfLifeSeconds);
}

double trendingDecayFactor(Timestamp updatedAt, Timestamp now, double halfLifeSeconds) {
    const double dt = now > updatedAt ? static_cast<double>(now - updatedAt) : 0.0;
    return halfLifeDecay(dt, halfLifeSeconds);
}

double trendingScore(const Reel &reel, Timestamp now, double halfLifeSeconds) {
    const double w = trendingDecayFactor(reel.trendingUpdatedAt, now, halfLifeSeconds);
    return w * reel.trendingEngagement / (1.0 + w * reel.trendingImpressions);
}

double engagementPriorMean(const std::vector<Reel> &reels) {
    double totalEngagement = 0.0;
    double totalImpressions = 0.0;
    for (const Reel &reel : reels) {
        totalEngagement += popularityEngagement(reel);
        totalImpressions += static_cast<double>(reel.impressionCount);
    }
    return totalImpressions > 0.0 ? totalEngagement / totalImpressions : 0.0;
}

} // namespace rr
