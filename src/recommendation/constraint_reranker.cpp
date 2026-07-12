#include "rr/recommendation/constraint_reranker.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rr/domain/ids.hpp"

namespace rr {

ConstraintReranker::ConstraintReranker(const std::vector<Reel> &reels,
                                       const DiversityConfig &config)
    : reels_(reels), config_(config) {}

std::size_t ConstraintReranker::topicCap(std::uint32_t maxPerTopic, std::size_t feedSize) {
    // The TDD states "three per ten-item feed"; scale that ceiling proportionally to the requested
    // feed size and floor at 1 so even a tiny feed (or a zero maxPerTopic) admits at least one.
    const double scaled =
        std::ceil(static_cast<double>(maxPerTopic) * static_cast<double>(feedSize) / 10.0);
    const std::size_t cap = static_cast<std::size_t>(scaled);
    return cap < 1 ? 1 : cap;
}

std::vector<Candidate>
ConstraintReranker::selectFeed(const User &user, const std::vector<Candidate> &rankedCandidates,
                               std::size_t feedSize) const {
    // Greedy walk in the GIVEN order (position == relevance); never re-sort. Admit a candidate iff
    // it violates no hard rule. Caps are hard — no relax/backfill pass — so the feed is shorter
    // than feedSize only when no remaining candidate is addable (this subsumes literal pool
    // exhaustion).
    const std::size_t topicLimit = topicCap(config_.maxPerTopic, feedSize);

    std::vector<Candidate> feed;
    feed.reserve(std::min(feedSize, rankedCandidates.size()));
    std::unordered_set<ReelId> chosen;
    std::unordered_map<CreatorId, std::uint32_t> creatorCounts;
    std::unordered_map<TopicId, std::size_t> topicCounts;

    for (const Candidate &c : rankedCandidates) {
        if (feed.size() >= feedSize) {
            break;
        }
        const ReelId id = c.reelId;
        // Out-of-range id: belt-and-braces (the Orchestrator already drops these). Treat as
        // ineligible so the dense-id lookup below is always safe.
        if (id.value >= reels_.size()) {
            continue;
        }
        if (chosen.contains(id)) {
            continue; // no duplicate reel ids
        }
        if (user.seenReels.contains(id)) {
            continue; // no reel the user has already seen
        }
        const Reel &reel = reels_[id.value];
        if (creatorCounts[reel.creatorId] >= config_.maxPerCreator) {
            continue; // <= maxPerCreator per creator
        }
        if (topicCounts[reel.primaryTopic] >= topicLimit) {
            continue; // <= topicCap per primary topic (scaled to feedSize)
        }
        // Admitted: record and keep the candidate verbatim (score/source/contributions preserved).
        chosen.insert(id);
        ++creatorCounts[reel.creatorId];
        ++topicCounts[reel.primaryTopic];
        feed.push_back(c);
    }
    return feed;
}

std::vector<RankedReel> ConstraintReranker::rerank(const User &user,
                                                   const std::vector<Candidate> &rankedCandidates,
                                                   std::size_t feedSize) const {
    std::vector<Candidate> feed = selectFeed(user, rankedCandidates, feedSize);

    // Consecutive-same-topic avoidance (TDD 15.1): a SINGLE greedy forward pass. When feed[i] and
    // feed[i+1] share a primary topic, swap feed[i+1] with the NEAREST later item whose topic
    // differs (unique smallest index => deterministic). Leave the run intact if none exists. Purely
    // cosmetic: the SET is unchanged. Provably terminating: each i is visited once with at most one
    // swap. Reel ids are all in range here (selectFeed dropped out-of-range candidates).
    auto topicOf = [this](const Candidate &c) { return reels_[c.reelId.value].primaryTopic; };
    for (std::size_t i = 0; i + 1 < feed.size(); ++i) {
        const TopicId t = topicOf(feed[i]);
        if (topicOf(feed[i + 1]) != t) {
            continue;
        }
        for (std::size_t j = i + 2; j < feed.size(); ++j) {
            if (topicOf(feed[j]) != t) {
                std::swap(feed[i + 1], feed[j]);
                break;
            }
        }
    }

    std::vector<RankedReel> out;
    out.reserve(feed.size());
    for (std::size_t i = 0; i < feed.size(); ++i) {
        Candidate &c = feed[i];
        RankedReel r{};
        r.reelId = c.reelId;
        r.score = c.rankingScore;
        r.rank = i;
        r.sources = {c.source};
        r.featureContributions = std::move(c.featureContributions);
        out.push_back(std::move(r));
    }
    return out;
}

} // namespace rr
