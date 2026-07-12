// Unit tests for the ConstraintReranker (Phase 9, TDD 15.1 / plan task 1). Each hard rule is
// exercised in isolation and combined; the documented short-feed behaviour, the consecutive-topic
// swap (including an unswappable all-same-topic case), the topic-cap scaling rule at feedSize !=
// 10, deterministic tie-breaking, and input-order-preservation are all asserted with hand-built
// pools.
#include "rr/recommendation/constraint_reranker.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"

namespace {

// Reels with dense ids; each carries a (creatorId, primaryTopic). Embeddings are irrelevant to the
// constraint reranker (it never reads them) but are non-empty for realism.
std::vector<rr::Reel> makeReels(const std::vector<std::pair<uint32_t, uint32_t>> &creatorTopic) {
    std::vector<rr::Reel> reels;
    reels.reserve(creatorTopic.size());
    for (std::size_t i = 0; i < creatorTopic.size(); ++i) {
        rr::Reel r{};
        r.id = rr::ReelId{static_cast<uint32_t>(i)};
        r.creatorId = rr::CreatorId{creatorTopic[i].first};
        r.primaryTopic = rr::TopicId{creatorTopic[i].second};
        r.embedding = {1.0f, 0.0f};
        r.active = true;
        reels.push_back(std::move(r));
    }
    return reels;
}

rr::Candidate cand(uint32_t id, float rankingScore = 1.0f,
                   rr::CandidateSource src = rr::CandidateSource::VectorHNSW) {
    rr::Candidate c{};
    c.reelId = rr::ReelId{id};
    c.source = src;
    c.rankingScore = rankingScore;
    return c;
}

rr::DiversityConfig cfg(uint32_t maxPerCreator = 2, uint32_t maxPerTopic = 3) {
    rr::DiversityConfig d{};
    d.maxPerCreator = maxPerCreator;
    d.maxPerTopic = maxPerTopic;
    return d;
}

std::vector<uint32_t> feedIds(const std::vector<rr::RankedReel> &feed) {
    std::vector<uint32_t> ids;
    for (const rr::RankedReel &r : feed) {
        ids.push_back(r.reelId.value);
    }
    return ids;
}

} // namespace

TEST(ConstraintRerankerTest, TopicCapScalingRule) {
    // The TDD's "three per ten-item feed" scaled proportionally: ceil(maxPerTopic * feedSize / 10),
    // floored at 1.
    EXPECT_EQ(rr::ConstraintReranker::topicCap(3, 10), 3u);
    EXPECT_EQ(rr::ConstraintReranker::topicCap(3, 20), 6u);
    EXPECT_EQ(rr::ConstraintReranker::topicCap(3, 30), 9u);
    EXPECT_EQ(rr::ConstraintReranker::topicCap(3, 5), 2u);  // ceil(1.5)
    EXPECT_EQ(rr::ConstraintReranker::topicCap(3, 1), 1u);  // ceil(0.3) -> floored to 1
    EXPECT_EQ(rr::ConstraintReranker::topicCap(0, 10), 1u); // floored to 1
}

TEST(ConstraintRerankerTest, RemovesDuplicateIds) {
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {1, 1}});
    rr::ConstraintReranker rr_(reels, cfg());
    rr::User user{};
    // reel 0 appears twice in the input; the second occurrence must be dropped as a duplicate id.
    const std::vector<rr::RankedReel> feed = rr_.rerank(user, {cand(0), cand(0), cand(1)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1}));
}

TEST(ConstraintRerankerTest, DropsSeenReels) {
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {1, 1}, {2, 2}});
    rr::ConstraintReranker rr_(reels, cfg());
    rr::User user{};
    user.seenReels.insert(rr::ReelId{1});
    const std::vector<rr::RankedReel> feed = rr_.rerank(user, {cand(0), cand(1), cand(2)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 2}));
}

TEST(ConstraintRerankerTest, EnforcesCreatorCap) {
    // reels 0,1,2 share creator 9; maxPerCreator = 2, so only the first two are admitted. reel 3
    // (creator 8) is admitted. Distinct topics keep the topic cap out of the way.
    std::vector<rr::Reel> reels = makeReels({{9, 0}, {9, 1}, {9, 2}, {8, 3}});
    rr::ConstraintReranker rr_(reels, cfg(/*maxPerCreator=*/2));
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        rr_.rerank(user, {cand(0), cand(1), cand(2), cand(3)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1, 3}));
}

TEST(ConstraintRerankerTest, EnforcesTopicCapAtFeedSizeTen) {
    // reels 0..3 share topic 7; at feedSize 10 the cap is 3, so reel 3 is dropped. Distinct
    // creators keep the creator cap out of the way. (All same topic => the swap pass leaves the
    // order intact.)
    std::vector<rr::Reel> reels = makeReels({{0, 7}, {1, 7}, {2, 7}, {3, 7}});
    rr::ConstraintReranker rr_(reels, cfg(/*maxPerCreator=*/2, /*maxPerTopic=*/3));
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        rr_.rerank(user, {cand(0), cand(1), cand(2), cand(3)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1, 2}));
}

TEST(ConstraintRerankerTest, TopicCapScalesWithFeedSize) {
    // Same topic-7 reels, but feedSize 20 => cap 6, so six of seven are admitted (distinct
    // creators).
    std::vector<rr::Reel> reels =
        makeReels({{0, 7}, {1, 7}, {2, 7}, {3, 7}, {4, 7}, {5, 7}, {6, 7}});
    rr::ConstraintReranker rr_(reels, cfg(/*maxPerCreator=*/2, /*maxPerTopic=*/3));
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        rr_.rerank(user, {cand(0), cand(1), cand(2), cand(3), cand(4), cand(5), cand(6)}, 20);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1, 2, 3, 4, 5}));
}

TEST(ConstraintRerankerTest, AvoidsConsecutiveSameTopicBySwap) {
    // Topics [A, A, B]: after selection the feed is [0(A), 1(A), 2(B)]; the swap pass breaks the
    // A,A run by swapping position 1 with the nearest later differing item (2), giving [0, 2, 1].
    std::vector<rr::Reel> reels = makeReels({{0, 100}, {1, 100}, {2, 200}});
    rr::ConstraintReranker rr_(reels, cfg(/*maxPerCreator=*/2, /*maxPerTopic=*/3));
    rr::User user{};
    const std::vector<rr::RankedReel> feed = rr_.rerank(user, {cand(0), cand(1), cand(2)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 2, 1}));
}

TEST(ConstraintRerankerTest, SwapLeavesUnswappableAllSameTopicIntact) {
    // Every admitted item shares a topic, so no differing later item exists to swap in: the feed is
    // left exactly in relevance order (documented "where possible" behaviour).
    std::vector<rr::Reel> reels = makeReels({{0, 5}, {1, 5}, {2, 5}});
    rr::ConstraintReranker rr_(reels, cfg(/*maxPerCreator=*/2, /*maxPerTopic=*/3));
    rr::User user{};
    const std::vector<rr::RankedReel> feed = rr_.rerank(user, {cand(0), cand(1), cand(2)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1, 2}));
}

TEST(ConstraintRerankerTest, ShortFeedWhenCreatorCapInfeasible) {
    // Five candidates all from creator 0; maxPerCreator 2 means only two can ever be admitted, so
    // the feed is SHORTER than the requested feedSize 5 — the documented cap-infeasible behaviour.
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}});
    rr::ConstraintReranker rr_(reels, cfg(/*maxPerCreator=*/2));
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        rr_.rerank(user, {cand(0), cand(1), cand(2), cand(3), cand(4)}, 5);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1}));
}

TEST(ConstraintRerankerTest, ShortFeedOnPoolExhaustion) {
    // Pool smaller than feedSize and fully compliant => the whole eligible pool is delivered.
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {1, 1}});
    rr::ConstraintReranker rr_(reels, cfg());
    rr::User user{};
    const std::vector<rr::RankedReel> feed = rr_.rerank(user, {cand(0), cand(1)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1}));
}

TEST(ConstraintRerankerTest, RespectsInputOrderNeverResorts) {
    // Input is deliberately NOT id-sorted; the feed must preserve the input's relevance order,
    // never re-sort to ascending id. (Distinct creators + topics so no cap or swap perturbs the
    // order.)
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {1, 1}, {2, 2}, {3, 3}});
    rr::ConstraintReranker rr_(reels, cfg());
    rr::User user{};
    const std::vector<rr::RankedReel> feed = rr_.rerank(user, {cand(3), cand(1), cand(2)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{3, 1, 2}));
}

TEST(ConstraintRerankerTest, CombinedConstraints) {
    // creators: [A,A,A,B,C]; topics: [T,T,U,V,W]; user has seen reel 4.
    // Walk: 0 admit (A,T); 1 admit (A,T); 2 SKIP (creator A already at cap 2); 3 admit (B,V);
    //       4 SKIP (seen). Selected set {0,1,3}. Swap: [0(T),1(T),3(V)] -> swap pos1 with 3 ->
    //       [0,3,1].
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {0, 0}, {0, 1}, {1, 2}, {2, 3}});
    rr::ConstraintReranker rr_(reels, cfg(/*maxPerCreator=*/2, /*maxPerTopic=*/3));
    rr::User user{};
    user.seenReels.insert(rr::ReelId{4});
    const std::vector<rr::RankedReel> feed =
        rr_.rerank(user, {cand(0), cand(1), cand(2), cand(3), cand(4)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 3, 1}));
}

TEST(ConstraintRerankerTest, PopulatesRankScoreSourceAndContributions) {
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {1, 1}});
    rr::ConstraintReranker rr_(reels, cfg());
    rr::User user{};
    rr::Candidate c0 = cand(0, 7.5f, rr::CandidateSource::Trending);
    c0.featureContributions = {{"quality", 0.4f}};
    rr::Candidate c1 = cand(1, 3.25f, rr::CandidateSource::Fresh);
    const std::vector<rr::RankedReel> feed = rr_.rerank(user, {c0, c1}, 10);
    ASSERT_EQ(feed.size(), 2u);
    EXPECT_EQ(feed[0].rank, 0u);
    EXPECT_EQ(feed[1].rank, 1u);
    EXPECT_FLOAT_EQ(feed[0].score, 7.5f);
    EXPECT_FLOAT_EQ(feed[1].score, 3.25f);
    ASSERT_EQ(feed[0].sources.size(), 1u);
    EXPECT_EQ(feed[0].sources.front(), rr::CandidateSource::Trending);
    EXPECT_EQ(feed[1].sources.front(), rr::CandidateSource::Fresh);
    ASSERT_EQ(feed[0].featureContributions.count("quality"), 1u);
    EXPECT_FLOAT_EQ(feed[0].featureContributions.at("quality"), 0.4f);
}

TEST(ConstraintRerankerTest, SelectFeedReturnsSetWithoutSwap) {
    // selectFeed exposes the chosen SET in relevance order WITHOUT the cosmetic swap: topics
    // [A,A,B] stay [0,1,2] here, whereas rerank() would swap to [0,2,1].
    std::vector<rr::Reel> reels = makeReels({{0, 100}, {1, 100}, {2, 200}});
    rr::ConstraintReranker rr_(reels, cfg());
    rr::User user{};
    const std::vector<rr::Candidate> set = rr_.selectFeed(user, {cand(0), cand(1), cand(2)}, 10);
    std::vector<uint32_t> ids;
    for (const rr::Candidate &c : set) {
        ids.push_back(c.reelId.value);
    }
    EXPECT_EQ(ids, (std::vector<uint32_t>{0, 1, 2}));
}

TEST(ConstraintRerankerTest, Deterministic) {
    std::vector<rr::Reel> reels = makeReels({{0, 0}, {0, 0}, {1, 1}, {2, 0}, {2, 2}});
    rr::ConstraintReranker rr_(reels, cfg());
    rr::User user{};
    const std::vector<rr::Candidate> pool = {cand(3), cand(0), cand(1), cand(4), cand(2)};
    EXPECT_EQ(feedIds(rr_.rerank(user, pool, 4)), feedIds(rr_.rerank(user, pool, 4)));
}
