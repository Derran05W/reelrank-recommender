#pragma once

#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/recommendation/feature_extractor.hpp"
#include "rr/recommendation/ranker.hpp"

namespace rr {

// TDD 14.2 / 14.4: the deterministic second-stage ranker. Scores each candidate by the weighted
// sum of its normalized features (weights from RankingConfig),
//
//   score = wS*S + wST*ST + wQ*Q + wF*F + wP*P + wT*T + wC*C + wE*E + wD*D - wR*R - wI*I
//
// where ST is the session-topic similarity (TDD 14.1), a positive contribution activated in
// Phase 7 alongside online session-vector updates (wST == session_topic_weight). Returns the
// candidates sorted by rankingScore DESCENDING, ties broken by ascending ReelId (a total order, so
// the output is fully deterministic - same doctrine as the Orchestrator).
//
// Every returned candidate carries its rankingScore AND a featureContributions map with all ELEVEN
// FROZEN snake_case keys always present, penalties stored as NEGATIVE values. The map's values
// sum to rankingScore to float tolerance (property-tested).
class WeightedRanker final : public Ranker {
  public:
    WeightedRanker(const std::vector<Reel> &reels, const RankingConfig &config);

    std::vector<Candidate> rank(const User &user, const std::vector<Candidate> &candidates,
                                Timestamp now) const override;

  private:
    RankingConfig config_;
    FeatureExtractor extractor_;
};

} // namespace rr
