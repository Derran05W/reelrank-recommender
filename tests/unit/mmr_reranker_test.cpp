// Unit tests for the MMRReranker (Phase 9, TDD 15.2 / plan task 2). Hand-built pools with
// hand-computed MMR selections at lambda in {0, 0.75, 1}, plus normalization edge cases, tie-breaks
// and determinism. Embeddings are 2-D unit vectors so the cosine (rr::dot) term is exact.
#include "rr/recommendation/mmr_reranker.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"

namespace {

std::vector<rr::Reel> makeReels(const std::vector<rr::Embedding> &embs) {
    std::vector<rr::Reel> reels;
    reels.reserve(embs.size());
    for (std::size_t i = 0; i < embs.size(); ++i) {
        rr::Reel r{};
        r.id = rr::ReelId{static_cast<uint32_t>(i)};
        r.embedding = embs[i];
        r.active = true;
        reels.push_back(std::move(r));
    }
    return reels;
}

rr::Candidate cand(uint32_t id, float rankingScore,
                   rr::CandidateSource src = rr::CandidateSource::VectorHNSW) {
    rr::Candidate c{};
    c.reelId = rr::ReelId{id};
    c.source = src;
    c.rankingScore = rankingScore;
    return c;
}

std::vector<uint32_t> feedIds(const std::vector<rr::RankedReel> &feed) {
    std::vector<uint32_t> ids;
    for (const rr::RankedReel &r : feed) {
        ids.push_back(r.reelId.value);
    }
    return ids;
}

} // namespace

TEST(MMRRerankerTest, LambdaOneIsPureRelevanceOrder) {
    // lambda == 1 drops the diversity term; the feed is descending rankingScore. Scores 0.2/0.9/0.5
    // for ids 0/1/2 => order 1, 2, 0. Embeddings are irrelevant here (weighted by 1 - lambda = 0).
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {0, 1}, {1, 0}});
    rr::MMRReranker mmr(reels, 1.0);
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        mmr.rerank(user, {cand(0, 0.2f), cand(1, 0.9f), cand(2, 0.5f)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{1, 2, 0}));
}

TEST(MMRRerankerTest, LambdaOneAllEqualScoresPreservesInputOrder) {
    // All scores equal => min-max relevance is undefined, so the position-based fallback (strictly
    // decreasing in input index) governs; at lambda == 1 that reproduces the INPUT order exactly.
    // Input is [id1, id0], so the feed must be [1, 0] (NOT id-sorted).
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {0, 1}});
    rr::MMRReranker mmr(reels, 1.0);
    rr::User user{};
    const std::vector<rr::RankedReel> feed = mmr.rerank(user, {cand(1, 0.5f), cand(0, 0.5f)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{1, 0}));
}

TEST(MMRRerankerTest, LambdaZeroIsPureDiversityAfterFirstPick) {
    // lambda == 0 drops relevance entirely; mmr = -maxSim. id0 and id1 are identical (cosine 1),
    // id2 is orthogonal (cosine 0). Equal scores => first pick ties on mmr(=0) and breaks to the
    // smallest id (0). Then id2 (penalty 0) beats id1 (penalty 1). Feed [0, 2, 1].
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {1, 0}, {0, 1}});
    rr::MMRReranker mmr(reels, 0.0);
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        mmr.rerank(user, {cand(0, 0.5f), cand(1, 0.5f), cand(2, 0.5f)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 2, 1}));
}

TEST(MMRRerankerTest, LambdaDefaultReordersForDiversity) {
    // lambda = 0.75. Embeddings: id0=id1=(1,0) (cosine 1), id2=(0,1) (cosine 0).
    // Scores 1.0/0.93/0.90
    // => min-max relevance 1.0/0.3/0.0.
    //   pick 1: mmr = 0.75*rel; id0 wins (0.75).
    //   pick 2: mmr(1) = 0.75*0.3 - 0.25*1 = -0.025; mmr(2) = 0.75*0 - 0.25*0 = 0 -> id2 wins.
    //   pick 3: id1. Feed [0, 2, 1] — the near-duplicate id1 is pushed BELOW the orthogonal id2,
    // which pure-relevance order [0, 1, 2] would not do.
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {1, 0}, {0, 1}});
    rr::MMRReranker mmr(reels, 0.75);
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        mmr.rerank(user, {cand(0, 1.0f), cand(1, 0.93f), cand(2, 0.90f)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 2, 1}));
}

TEST(MMRRerankerTest, FirstPickTieBreaksByHigherScore) {
    // lambda == 0 => every first-pick mmr is 0; the tie breaks to the HIGHER rankingScore. Input is
    // [id0(0.5), id1(0.7)]; id1 (higher score) is picked first despite id0 appearing first.
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {0, 1}});
    rr::MMRReranker mmr(reels, 0.0);
    rr::User user{};
    const std::vector<rr::RankedReel> feed = mmr.rerank(user, {cand(0, 0.5f), cand(1, 0.7f)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{1, 0}));
}

TEST(MMRRerankerTest, TieBreaksByAscendingIdWhenScoresEqual) {
    // lambda == 0, equal scores => first-pick mmr and score both tie, so the smallest id wins even
    // though the input leads with id1. Feed [0, 1].
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {0, 1}});
    rr::MMRReranker mmr(reels, 0.0);
    rr::User user{};
    const std::vector<rr::RankedReel> feed = mmr.rerank(user, {cand(1, 0.5f), cand(0, 0.5f)}, 10);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1}));
}

TEST(MMRRerankerTest, SelectsUpToFeedSize) {
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {0, 1}, {1, 0}, {0, 1}});
    rr::MMRReranker mmr(reels, 1.0);
    rr::User user{};
    const std::vector<rr::RankedReel> feed =
        mmr.rerank(user, {cand(0, 0.9f), cand(1, 0.8f), cand(2, 0.7f), cand(3, 0.6f)}, 2);
    EXPECT_EQ(feed.size(), 2u);
    EXPECT_EQ(feedIds(feed), (std::vector<uint32_t>{0, 1})); // top two by relevance at lambda 1
}

TEST(MMRRerankerTest, PopulatesRankScoreSourceAndContributions) {
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {0, 1}});
    rr::MMRReranker mmr(reels, 1.0);
    rr::User user{};
    rr::Candidate c0 = cand(0, 0.9f, rr::CandidateSource::Trending);
    c0.featureContributions = {{"quality", 0.4f}};
    rr::Candidate c1 = cand(1, 0.4f, rr::CandidateSource::Fresh);
    const std::vector<rr::RankedReel> feed = mmr.rerank(user, {c0, c1}, 10);
    ASSERT_EQ(feed.size(), 2u);
    EXPECT_EQ(feed[0].rank, 0u);
    EXPECT_EQ(feed[1].rank, 1u);
    EXPECT_FLOAT_EQ(feed[0].score, 0.9f); // score is the untouched rankingScore
    EXPECT_FLOAT_EQ(feed[1].score, 0.4f);
    ASSERT_EQ(feed[0].sources.size(), 1u);
    EXPECT_EQ(feed[0].sources.front(), rr::CandidateSource::Trending);
    ASSERT_EQ(feed[0].featureContributions.count("quality"), 1u);
    EXPECT_FLOAT_EQ(feed[0].featureContributions.at("quality"), 0.4f);
}

TEST(MMRRerankerTest, Deterministic) {
    std::vector<rr::Reel> reels = makeReels({{1, 0}, {1, 0}, {0, 1}, {1, 0}});
    rr::MMRReranker mmr(reels, 0.75);
    rr::User user{};
    const std::vector<rr::Candidate> pool = {cand(0, 1.0f), cand(1, 0.93f), cand(2, 0.9f),
                                             cand(3, 0.8f)};
    EXPECT_EQ(feedIds(mmr.rerank(user, pool, 4)), feedIds(mmr.rerank(user, pool, 4)));
}
