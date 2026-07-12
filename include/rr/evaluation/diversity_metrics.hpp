#pragma once

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"

namespace rr {

// Per-feed diversity measurement (TDD 18.4, Phase 9). Six values computed from ONE recommendation
// feed, the dense reel catalog, and the requesting user's seen-set AS OF presentation time. The
// calculator is PURE: rng-free, wall-clock-free, and it reads only evaluation-visible state without
// mutating anything the recommender or simulator sees (TDD 18.2 evaluation carve-out). Every metric
// definition is documented at its field below and re-stated at computeFeedDiversity's definition.
//
// This measures the diversity of ANY algorithm's output: it is the measurement side of Phase 9 and
// deliberately does not depend on the rerankers. Cost is trivial (a feedSize-10 feed is 45 dot
// products), so the harness runs it on EVERY request, unsampled (unlike the oracle/retrieval
// probes) - there is no reason to sample a measurement this cheap and diversity should be reported
// for every feed, not a subset.
struct FeedDiversity {
    // Number of items in the feed the metrics were computed over.
    std::size_t feedSize = 0;

    // Count of DISTINCT primaryTopic values among the feed's items. Range [0, feedSize]; higher is
    // more topically diverse.
    std::size_t uniqueTopics = 0;

    // Count of DISTINCT creatorId values among the feed's items. Range [0, feedSize]; higher is
    // more creator-diverse.
    std::size_t uniqueCreators = 0;

    // Mean pairwise embedding cosine over ALL unordered item pairs (rr::dot; embeddings are
    // unit-length so dot == cosine, D3). DEFINED as 0.0 for feeds with fewer than 2 items (no pairs
    // exist). Range [-1, 1]; LOWER means a more spread-out (less self-similar) feed.
    double intraListSimilarity = 0.0;

    // Herfindahl-Hirschman index over primary-topic shares: sum over distinct topics of
    // (count/feedSize)^2. Equals 1.0 when every item shares one topic; equals 1/k when the feed is
    // spread uniformly over k distinct topics (so 1/feedSize when every item's topic is distinct).
    // DEFINED as 0.0 for an empty feed (no shares). Range (0, 1] for a non-empty feed; LOWER means
    // less topic concentration.
    double topicConcentration = 0.0;

    // Herfindahl-Hirschman index over creator shares (same definition/range as topicConcentration,
    // over creatorId). DEFINED as 0.0 for an empty feed.
    double creatorConcentration = 0.0;

    // Number of feed items that are REPEATS. An item is a repeat when it is EITHER already in the
    // user's seen-set at presentation time (shown earlier in this run) OR a duplicate of an
    // earlier-positioned item within this same feed. Both conditions are structurally 0 in the
    // current pipeline (the orchestrator seen-filters and de-duplicates), so this is published as a
    // LIVE verification of the Phase 9 "duplicate/repetitive content eliminated" exit criterion.
    std::size_t repeatCount = 0;

    // repeatCount / feedSize; the fraction of the feed that is repeats. DEFINED as 0.0 for an empty
    // feed. Range [0, 1]; expected 0 by construction (see repeatCount).
    double repetitionRate = 0.0;
};

// Aggregate diversity over a SET of feeds (one round, or a whole run). The per-feed metrics are
// averaged over feeds; repetition is pooled as total repeats over total items so it is a true rate
// even when feeds differ in length. All means are 0 when no feeds were aggregated.
struct DiversitySummary {
    std::size_t feeds = 0; // number of feeds aggregated

    double meanUniqueTopics = 0.0;         // mean FeedDiversity::uniqueTopics over feeds
    double meanUniqueCreators = 0.0;       // mean FeedDiversity::uniqueCreators over feeds
    double meanIntraListSimilarity = 0.0;  // mean FeedDiversity::intraListSimilarity over feeds
    double meanTopicConcentration = 0.0;   // mean FeedDiversity::topicConcentration over feeds
    double meanCreatorConcentration = 0.0; // mean FeedDiversity::creatorConcentration over feeds

    std::size_t totalRepeats = 0; // total repeat items over all aggregated feeds (expected 0)
    std::size_t totalItems = 0;   // total feed items over all aggregated feeds
    double repetitionRate = 0.0;  // totalRepeats / totalItems (0.0 when totalItems == 0)
};

// PURE per-feed calculator (TDD 18.4). `feedReelIds` is the ordered list of reel ids in the feed;
// `reels` is the dense catalog (indexed by ReelId::value, TDD 8.2); `seenReels` is the user's
// seen-set AS OF presentation time (before this feed's impressions are stepped). Consumes no rng
// and no clock and mutates nothing. Definitions (restated from FeedDiversity):
//   uniqueTopics/uniqueCreators : distinct primaryTopic / creatorId counts.
//   intraListSimilarity         : mean cosine over all unordered pairs (0 for < 2 items).
//   topicConcentration          : HHI = sum of squared primary-topic shares (0 for empty feed).
//   creatorConcentration        : HHI over creator shares.
//   repeatCount/repetitionRate  : items seen earlier this run OR duplicated within the feed.
FeedDiversity computeFeedDiversity(const std::vector<ReelId> &feedReelIds,
                                   const std::vector<Reel> &reels,
                                   const std::unordered_set<ReelId> &seenReels);

// Small per-round + overall accumulator over feeds, analogous to the runner's RegretAcc /
// RetrievalAcc pattern but living in this module so it is independently unit-testable with
// hand-built feeds. Deterministic and rng-free.
class DiversityAccumulator {
  public:
    // One accumulator slot per round. add() with an out-of-range round is a no-op-safe assert in
    // Debug (the harness always passes a valid round).
    explicit DiversityAccumulator(std::size_t rounds);

    // Fold one already-computed feed's metrics into `round`.
    void add(std::size_t round, const FeedDiversity &feed);

    // Convenience: compute the feed's metrics (computeFeedDiversity) and fold them into `round`.
    void add(std::size_t round, const std::vector<ReelId> &feedReelIds,
             const std::vector<Reel> &reels, const std::unordered_set<ReelId> &seenReels);

    // Mean-over-feeds summary for one round (empty summary if the round has no feeds).
    DiversitySummary roundSummary(std::size_t round) const;

    // Mean-over-feeds summary pooled over ALL rounds. Pools the grand sums (not a mean of per-round
    // means), so it is the exact mean over every feed in the run.
    DiversitySummary overall() const;

    std::size_t roundCount() const { return rounds_.size(); }

  private:
    // Running sums for one round; means are formed lazily in roundSummary/overall.
    struct RoundAcc {
        std::size_t feeds = 0;
        double uniqueTopicsSum = 0.0;
        double uniqueCreatorsSum = 0.0;
        double intraListSimilaritySum = 0.0;
        double topicConcentrationSum = 0.0;
        double creatorConcentrationSum = 0.0;
        std::size_t totalRepeats = 0;
        std::size_t totalItems = 0;
    };

    static DiversitySummary summarize(const RoundAcc &acc);

    std::vector<RoundAcc> rounds_;
};

} // namespace rr
