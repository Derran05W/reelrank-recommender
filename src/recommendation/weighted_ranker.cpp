#include "rr/recommendation/weighted_ranker.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rr/recommendation/feature_extractor.hpp"

namespace rr {

WeightedRanker::WeightedRanker(const std::vector<Reel> &reels, const RankingConfig &config)
    : config_(config), extractor_(reels, config) {}

std::vector<Candidate> WeightedRanker::rank(const User &user,
                                            const std::vector<Candidate> &candidates,
                                            Timestamp now) const {
    const std::vector<FeatureVector> features = extractor_.extract(user, candidates, now);

    std::vector<Candidate> ranked = candidates;
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        const FeatureVector &f = features[i];

        // TDD 14.2 weighted contributions. Penalties (repetition, impression fatigue) are stored
        // as NEGATIVE values so the map sums to the score directly. All ten FROZEN snake_case keys
        // are ALWAYS present. Computed in double, then stored as float; rankingScore is the sum of
        // the stored float contributions, so the map's values sum to rankingScore to within one
        // float rounding (property-tested).
        const float similarity = static_cast<float>(config_.similarityWeight * f.similarity);
        const float quality = static_cast<float>(config_.qualityWeight * f.quality);
        const float freshness = static_cast<float>(config_.freshnessWeight * f.freshness);
        const float popularity = static_cast<float>(config_.popularityWeight * f.popularity);
        const float trending = static_cast<float>(config_.trendingWeight * f.trending);
        const float creatorAffinity =
            static_cast<float>(config_.creatorAffinityWeight * f.creatorAffinity);
        const float exploration = static_cast<float>(config_.explorationWeight * f.exploration);
        const float durationMatch =
            static_cast<float>(config_.durationMatchWeight * f.durationMatch);
        const float repetitionPenalty =
            static_cast<float>(-config_.repetitionPenalty * f.repetition);
        const float impressionPenalty =
            static_cast<float>(-config_.impressionPenaltyWeight * f.impressionCount);

        const double sum = static_cast<double>(similarity) + quality + freshness + popularity +
                           trending + creatorAffinity + exploration + durationMatch +
                           repetitionPenalty + impressionPenalty;

        Candidate &c = ranked[i];
        c.rankingScore = static_cast<float>(sum);
        c.featureContributions = {
            {"similarity", similarity},
            {"quality", quality},
            {"freshness", freshness},
            {"popularity", popularity},
            {"trending", trending},
            {"creator_affinity", creatorAffinity},
            {"exploration", exploration},
            {"duration_match", durationMatch},
            {"repetition_penalty", repetitionPenalty},
            {"impression_penalty", impressionPenalty},
        };
    }

    // Sort by rankingScore DESCENDING, ties by ascending ReelId. ReelIds are unique in a
    // deduplicated pool, so this is a TOTAL order and the output is fully deterministic.
    std::sort(ranked.begin(), ranked.end(), [](const Candidate &a, const Candidate &b) {
        if (a.rankingScore != b.rankingScore) {
            return a.rankingScore > b.rankingScore;
        }
        return a.reelId.value < b.reelId.value;
    });

    return ranked;
}

} // namespace rr
