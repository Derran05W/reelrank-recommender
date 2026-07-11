#pragma once

#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"

namespace rr {

// The normalized feature values for one candidate (TDD 14.1). Every field is DETERMINISTIC and
// normalized into [0, 1]; the ranker applies the per-feature weight and (for the two penalties)
// the sign. The `repetition` and `impressionCount` fields are the raw [0,1] penalty magnitudes —
// they enter the score as NEGATIVE contributions, but the feature value itself is non-negative.
struct FeatureVector {
    float similarity;      // (cosine + 1) / 2
    float sessionTopic;    // (cos(user.sessionPreference, embedding) + 1) / 2  (TDD 14.1)
    float quality;         // reel.intrinsicQuality (already [0,1])
    float freshness;       // freshnessScore(createdAt, now, halfLife)
    float popularity;      // pool min-max of smoothedPopularity
    float trending;        // raw / (raw + 1) saturation of trendingScore
    float creatorAffinity; // user.creatorAffinity lookup (absent => 0)
    float exploration;     // constant 0.0 until Phase 8
    float durationMatch;   // 1 - |candDuration - preferredDuration| / durationRange
    float repetition;      // fraction of recent window sharing creator/topic (penalty magnitude)
    float impressionCount; // log-normalized global impressionCount (fatigue penalty magnitude)
};

// Computes the normalized ranking-feature vector for every candidate in a pool (TDD 14.1/14.3).
//
// Determinism (D8): pure, no randomness. Time enters only through the `now` Timestamp (D9).
// Performance: one call is O(pool * recentWindow) total and touches NOTHING O(catalog) — in
// particular the global engagementPriorMean(all reels) is banned here; the popularity prior is
// computed POOL-LOCALLY. Pool-relative features (popularity) are computed over the pool, so the
// extractor works on the whole pool at once rather than one candidate at a time.
//
// Every normalization rule is documented at its definition in feature_extractor.cpp; that is a
// Phase 6 exit criterion.
class FeatureExtractor {
  public:
    // Fixed reference scale for the impression-count log-normalization: an impression count of
    // this many maps to ~1.0 (documented in feature_extractor.cpp).
    static constexpr double kImpressionLogScale = 100000.0;

    // The generator's full duration span (5-120 s, reel_generator.cpp): the denominator of the
    // duration-match normalization.
    static constexpr double kDurationRangeSeconds = 115.0;

    FeatureExtractor(const std::vector<Reel> &reels, const RankingConfig &config);

    std::vector<FeatureVector> extract(const User &user, const std::vector<Candidate> &pool,
                                       Timestamp now) const;

  private:
    const std::vector<Reel> &reels_;
    RankingConfig config_;
};

} // namespace rr
