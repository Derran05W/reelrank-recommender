// Unit tests for the composite DiversityReranker (Phase 9, plan task 2 composition). Verifies the
// documented composition: useMmr == false reproduces ConstraintReranker exactly; useMmr == true
// keeps the SAME feed SET the constraints pick but orders it with MMR; ranks stay contiguous.
#include "rr/recommendation/diversity_reranker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/recommendation/constraint_reranker.hpp"
#include "rr/recommendation/mmr_reranker.hpp"

namespace {

// Reels with dense ids carrying (creator, topic, embedding).
struct Spec {
    uint32_t creator;
    uint32_t topic;
    rr::Embedding embedding;
};

std::vector<rr::Reel> makeReels(const std::vector<Spec> &specs) {
    std::vector<rr::Reel> reels;
    reels.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        rr::Reel r{};
        r.id = rr::ReelId{static_cast<uint32_t>(i)};
        r.creatorId = rr::CreatorId{specs[i].creator};
        r.primaryTopic = rr::TopicId{specs[i].topic};
        r.embedding = specs[i].embedding;
        r.active = true;
        reels.push_back(std::move(r));
    }
    return reels;
}

rr::Candidate cand(uint32_t id, float rankingScore) {
    rr::Candidate c{};
    c.reelId = rr::ReelId{id};
    c.source = rr::CandidateSource::VectorHNSW;
    c.rankingScore = rankingScore;
    return c;
}

rr::DiversityConfig cfg(bool useMmr) {
    rr::DiversityConfig d{};
    d.maxPerCreator = 2;
    d.maxPerTopic = 3;
    d.mmrLambda = 0.75;
    d.useMmr = useMmr;
    return d;
}

std::vector<uint32_t> feedIds(const std::vector<rr::RankedReel> &feed) {
    std::vector<uint32_t> ids;
    for (const rr::RankedReel &r : feed) {
        ids.push_back(r.reelId.value);
    }
    return ids;
}

std::vector<uint32_t> sortedIds(const std::vector<rr::RankedReel> &feed) {
    std::vector<uint32_t> ids = feedIds(feed);
    std::sort(ids.begin(), ids.end());
    return ids;
}

void expectContiguousRanks(const std::vector<rr::RankedReel> &feed) {
    for (std::size_t i = 0; i < feed.size(); ++i) {
        EXPECT_EQ(feed[i].rank, i);
    }
}

} // namespace

TEST(DiversityRerankerTest, UseMmrFalseIsIdenticalToConstraintReranker) {
    // Topics [A, A, B] trigger the constraint reranker's consecutive-topic swap (-> [0, 2, 1]);
    // with useMmr == false the composite must reproduce ConstraintReranker.rerank byte-for-byte.
    std::vector<rr::Reel> reels = makeReels({{0, 100, {1, 0}}, {1, 100, {1, 0}}, {2, 200, {0, 1}}});
    const std::vector<rr::Candidate> pool = {cand(0, 0.9f), cand(1, 0.8f), cand(2, 0.7f)};
    rr::User user{};

    rr::ConstraintReranker constraint(reels, cfg(false));
    rr::DiversityReranker diversity(reels, cfg(false));
    const std::vector<rr::RankedReel> expected = constraint.rerank(user, pool, 10);
    const std::vector<rr::RankedReel> actual = diversity.rerank(user, pool, 10);

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i].reelId, expected[i].reelId);
        EXPECT_FLOAT_EQ(actual[i].score, expected[i].score);
        EXPECT_EQ(actual[i].rank, expected[i].rank);
    }
    EXPECT_EQ(feedIds(actual), (std::vector<uint32_t>{0, 2, 1}));
}

TEST(DiversityRerankerTest, UseMmrTrueKeepsConstraintSetOrderedByMmr) {
    // Distinct topics => the constraint stage admits all three in relevance order [0, 1, 2] with NO
    // swap. With useMmr == true the SET is unchanged but ORDER is MMR's: id0=id1=(1,0) are cosine-
    // identical, id2=(0,1) orthogonal; scores 1.0/0.93/0.90 => MMR pushes near-duplicate id1 below
    // orthogonal id2 => [0, 2, 1], which differs from the useMmr==false order [0, 1, 2].
    std::vector<rr::Reel> reels = makeReels({{0, 10, {1, 0}}, {1, 20, {1, 0}}, {2, 30, {0, 1}}});
    const std::vector<rr::Candidate> pool = {cand(0, 1.0f), cand(1, 0.93f), cand(2, 0.90f)};
    rr::User user{};

    rr::ConstraintReranker constraint(reels, cfg(true));
    rr::MMRReranker mmr(reels, 0.75);
    rr::DiversityReranker diversity(reels, cfg(true));

    const std::vector<rr::RankedReel> constraintFeed = constraint.rerank(user, pool, 10);
    const std::vector<rr::RankedReel> actual = diversity.rerank(user, pool, 10);

    // Same SET as the constraints select...
    EXPECT_EQ(sortedIds(actual), sortedIds(constraintFeed));
    // ...ordered exactly as MMR orders that very set...
    const std::vector<rr::Candidate> set = constraint.selectFeed(user, pool, 10);
    const std::vector<rr::RankedReel> mmrOfSet = mmr.rerank(user, set, set.size());
    EXPECT_EQ(feedIds(actual), feedIds(mmrOfSet));
    // ...which here is a genuine reorder relative to the constraints' own (no-swap) order.
    EXPECT_EQ(feedIds(actual), (std::vector<uint32_t>{0, 2, 1}));
    EXPECT_EQ(feedIds(constraintFeed), (std::vector<uint32_t>{0, 1, 2}));
}

TEST(DiversityRerankerTest, RanksContiguousInBothModes) {
    std::vector<rr::Reel> reels =
        makeReels({{0, 10, {1, 0}}, {1, 20, {1, 0}}, {2, 30, {0, 1}}, {3, 40, {0, 1}}});
    const std::vector<rr::Candidate> pool = {cand(0, 1.0f), cand(1, 0.93f), cand(2, 0.90f),
                                             cand(3, 0.85f)};
    rr::User user{};
    expectContiguousRanks(rr::DiversityReranker(reels, cfg(false)).rerank(user, pool, 10));
    expectContiguousRanks(rr::DiversityReranker(reels, cfg(true)).rerank(user, pool, 10));
}

TEST(DiversityRerankerTest, EnforcesHardCapsInBothModes) {
    // Five candidates all from creator 0 (cap 2) => both modes deliver exactly the same 2-item SET.
    std::vector<rr::Reel> reels =
        makeReels({{0, 0, {1, 0}}, {0, 1, {0, 1}}, {0, 2, {1, 0}}, {0, 3, {0, 1}}, {0, 4, {1, 0}}});
    const std::vector<rr::Candidate> pool = {cand(0, 1.0f), cand(1, 0.9f), cand(2, 0.8f),
                                             cand(3, 0.7f), cand(4, 0.6f)};
    rr::User user{};
    EXPECT_EQ(sortedIds(rr::DiversityReranker(reels, cfg(false)).rerank(user, pool, 5)),
              (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(sortedIds(rr::DiversityReranker(reels, cfg(true)).rerank(user, pool, 5)),
              (std::vector<uint32_t>{0, 1}));
}
