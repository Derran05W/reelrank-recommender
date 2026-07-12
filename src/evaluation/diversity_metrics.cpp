#include "rr/evaluation/diversity_metrics.hpp"

#include <cassert>
#include <cstddef>
#include <unordered_map>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"

namespace rr {

FeedDiversity computeFeedDiversity(const std::vector<ReelId> &feedReelIds,
                                   const std::vector<Reel> &reels,
                                   const std::unordered_set<ReelId> &seenReels) {
    FeedDiversity d;
    const std::size_t n = feedReelIds.size();
    d.feedSize = n;

    // Single pass over the feed: tally topic/creator shares and classify repeats. `withinFeedSeen`
    // is the set of ids already emitted earlier in THIS feed, so a second occurrence of the same id
    // is flagged as a within-feed duplicate (the first occurrence is not).
    std::unordered_map<TopicId, std::size_t> topicCounts;
    std::unordered_map<CreatorId, std::size_t> creatorCounts;
    std::unordered_set<ReelId> withinFeedSeen;
    withinFeedSeen.reserve(n);
    std::size_t repeats = 0;

    for (const ReelId reelId : feedReelIds) {
        const Reel &reel = reels[reelId.value];
        ++topicCounts[reel.primaryTopic];
        ++creatorCounts[reel.creatorId];

        // Repeat = shown to this user earlier in the run (seen-set at presentation time) OR a
        // duplicate of an earlier position within this same feed. Both are expected 0 given the
        // orchestrator's seen-filter + dedup; the metric is the live verification of that.
        const bool seenEarlier = seenReels.contains(reelId);
        const bool duplicateInFeed = withinFeedSeen.contains(reelId);
        if (seenEarlier || duplicateInFeed) {
            ++repeats;
        }
        withinFeedSeen.insert(reelId);
    }

    d.uniqueTopics = topicCounts.size();
    d.uniqueCreators = creatorCounts.size();
    d.repeatCount = repeats;
    d.repetitionRate = n > 0 ? static_cast<double>(repeats) / static_cast<double>(n) : 0.0;

    // HHI = sum of squared shares. For an empty feed there are no shares, so both concentrations
    // stay at their 0.0 defaults (documented). Otherwise the share is count/feedSize.
    if (n > 0) {
        const double denom = static_cast<double>(n);
        double topicHHI = 0.0;
        for (const auto &[topic, count] : topicCounts) {
            const double share = static_cast<double>(count) / denom;
            topicHHI += share * share;
        }
        double creatorHHI = 0.0;
        for (const auto &[creator, count] : creatorCounts) {
            const double share = static_cast<double>(count) / denom;
            creatorHHI += share * share;
        }
        d.topicConcentration = topicHHI;
        d.creatorConcentration = creatorHHI;
    }

    // Intra-list similarity: mean cosine over all unordered pairs. Embeddings are unit-length, so
    // rr::dot IS the cosine (D3). Defined 0.0 for feeds with < 2 items (no pairs); the sum is
    // accumulated in double for stability.
    if (n >= 2) {
        double simSum = 0.0;
        std::size_t pairs = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const Embedding &ei = reels[feedReelIds[i].value].embedding;
            for (std::size_t j = i + 1; j < n; ++j) {
                simSum += dot(ei, reels[feedReelIds[j].value].embedding);
                ++pairs;
            }
        }
        d.intraListSimilarity = pairs > 0 ? simSum / static_cast<double>(pairs) : 0.0;
    }

    return d;
}

DiversityAccumulator::DiversityAccumulator(std::size_t rounds) : rounds_(rounds) {}

void DiversityAccumulator::add(std::size_t round, const FeedDiversity &feed) {
    assert(round < rounds_.size() && "DiversityAccumulator::add round out of range");
    if (round >= rounds_.size()) {
        return; // defensive in Release: never fold a stray feed into a nonexistent round
    }
    RoundAcc &acc = rounds_[round];
    acc.feeds += 1;
    acc.uniqueTopicsSum += static_cast<double>(feed.uniqueTopics);
    acc.uniqueCreatorsSum += static_cast<double>(feed.uniqueCreators);
    acc.intraListSimilaritySum += feed.intraListSimilarity;
    acc.topicConcentrationSum += feed.topicConcentration;
    acc.creatorConcentrationSum += feed.creatorConcentration;
    acc.totalRepeats += feed.repeatCount;
    acc.totalItems += feed.feedSize;
}

void DiversityAccumulator::add(std::size_t round, const std::vector<ReelId> &feedReelIds,
                               const std::vector<Reel> &reels,
                               const std::unordered_set<ReelId> &seenReels) {
    add(round, computeFeedDiversity(feedReelIds, reels, seenReels));
}

DiversitySummary DiversityAccumulator::summarize(const RoundAcc &acc) {
    DiversitySummary s;
    s.feeds = acc.feeds;
    s.totalRepeats = acc.totalRepeats;
    s.totalItems = acc.totalItems;
    if (acc.feeds > 0) {
        const double f = static_cast<double>(acc.feeds);
        s.meanUniqueTopics = acc.uniqueTopicsSum / f;
        s.meanUniqueCreators = acc.uniqueCreatorsSum / f;
        s.meanIntraListSimilarity = acc.intraListSimilaritySum / f;
        s.meanTopicConcentration = acc.topicConcentrationSum / f;
        s.meanCreatorConcentration = acc.creatorConcentrationSum / f;
    }
    s.repetitionRate = acc.totalItems > 0 ? static_cast<double>(acc.totalRepeats) /
                                                static_cast<double>(acc.totalItems)
                                          : 0.0;
    return s;
}

DiversitySummary DiversityAccumulator::roundSummary(std::size_t round) const {
    if (round >= rounds_.size()) {
        return DiversitySummary{};
    }
    return summarize(rounds_[round]);
}

DiversitySummary DiversityAccumulator::overall() const {
    // Pool the grand sums so the result is the exact mean over EVERY feed in the run (not a mean of
    // per-round means, which would be wrong when rounds hold different feed counts).
    RoundAcc grand;
    for (const RoundAcc &acc : rounds_) {
        grand.feeds += acc.feeds;
        grand.uniqueTopicsSum += acc.uniqueTopicsSum;
        grand.uniqueCreatorsSum += acc.uniqueCreatorsSum;
        grand.intraListSimilaritySum += acc.intraListSimilaritySum;
        grand.topicConcentrationSum += acc.topicConcentrationSum;
        grand.creatorConcentrationSum += acc.creatorConcentrationSum;
        grand.totalRepeats += acc.totalRepeats;
        grand.totalItems += acc.totalItems;
    }
    return summarize(grand);
}

} // namespace rr
