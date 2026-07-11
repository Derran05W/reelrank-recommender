// Unit tests for the WeightedRanker (Phase 6, TDD 14.2/14.4). Small fixtures with known counters,
// affinities, and history make the weighted-sum score, every contribution, weight plumbing, and
// the tie-break exactly hand-computable.
#include "rr/recommendation/weighted_ranker.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"

namespace {

rr::Reel makeReel(uint32_t id, uint32_t creator = 0, uint32_t topic = 0) {
    rr::Reel r{};
    r.id = rr::ReelId{id};
    r.creatorId = rr::CreatorId{creator};
    r.primaryTopic = rr::TopicId{topic};
    r.embedding = {1.0f, 0.0f};
    r.intrinsicQuality = 0.5f;
    r.durationSeconds = 30.0f;
    r.createdAt = 0;
    r.active = true;
    return r;
}

rr::Candidate cand(uint32_t id, float similarity = 0.0f) {
    rr::Candidate c{};
    c.reelId = rr::ReelId{id};
    c.source = rr::CandidateSource::VectorHNSW;
    c.retrievalSimilarity = similarity;
    return c;
}

rr::InteractionEvent event(uint32_t reelId, uint32_t creator, rr::InteractionType type) {
    rr::InteractionEvent e{};
    e.reelId = rr::ReelId{reelId};
    e.creatorId = rr::CreatorId{creator};
    e.type = type;
    return e;
}

const rr::Candidate &findById(const std::vector<rr::Candidate> &ranked, uint32_t id) {
    for (const rr::Candidate &c : ranked) {
        if (c.reelId.value == id) {
            return c;
        }
    }
    ADD_FAILURE() << "reel " << id << " missing from ranker output";
    return ranked.front();
}

} // namespace

TEST(WeightedRankerTest, HandComputedScoreAndContributions) {
    // Single-candidate pool (popularity is degenerate => 0.5).
    //  similarity: (0.6+1)/2 = 0.8   -> 0.50 * 0.8 = 0.40
    //  quality:    0.5               -> 0.10 * 0.5 = 0.05
    //  freshness:  createdAt==now=>1 -> 0.08 * 1.0 = 0.08
    //  popularity: 0.5 (degenerate)  -> 0.07 * 0.5 = 0.035
    //  trending:   raw=1 => 0.5      -> 0.08 * 0.5 = 0.04
    //  affinity:   0.6               -> 0.07 * 0.6 = 0.042
    //  exploration:0                 -> 0
    //  duration:   no history => 0.5 -> 0.05 * 0.5 = 0.025
    //  repetition: 0                 -> 0
    //  impression: 0                 -> 0
    //  score = 0.672
    std::vector<rr::Reel> reels{makeReel(0, /*creator=*/7)};
    reels[0].intrinsicQuality = 0.5f;
    reels[0].trendingUpdatedAt = 100;
    reels[0].trendingEngagement = 1.0;
    reels[0].trendingImpressions = 0.0;
    rr::User user{};
    user.creatorAffinity[rr::CreatorId{7}] = 0.6f;

    rr::WeightedRanker ranker(reels, {});
    const std::vector<rr::Candidate> ranked =
        ranker.rank(user, std::vector<rr::Candidate>{cand(0, 0.6f)}, 100);
    ASSERT_EQ(ranked.size(), 1u);
    const rr::Candidate &c = ranked.front();
    EXPECT_NEAR(c.rankingScore, 0.672f, 1e-5);

    const auto &m = c.featureContributions;
    EXPECT_NEAR(m.at("similarity"), 0.40f, 1e-5);
    EXPECT_NEAR(m.at("quality"), 0.05f, 1e-5);
    EXPECT_NEAR(m.at("freshness"), 0.08f, 1e-5);
    EXPECT_NEAR(m.at("popularity"), 0.035f, 1e-5);
    EXPECT_NEAR(m.at("trending"), 0.04f, 1e-5);
    EXPECT_NEAR(m.at("creator_affinity"), 0.042f, 1e-5);
    EXPECT_NEAR(m.at("exploration"), 0.0f, 1e-5);
    EXPECT_NEAR(m.at("duration_match"), 0.025f, 1e-5);
    EXPECT_NEAR(m.at("repetition_penalty"), 0.0f, 1e-5);
    EXPECT_NEAR(m.at("impression_penalty"), 0.0f, 1e-5);
}

TEST(WeightedRankerTest, AllTenKeysAlwaysPresent) {
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};
    rr::WeightedRanker ranker(reels, {});
    const std::vector<rr::Candidate> ranked =
        ranker.rank(user, std::vector<rr::Candidate>{cand(0)}, 0);
    ASSERT_EQ(ranked.size(), 1u);
    const auto &m = ranked.front().featureContributions;
    EXPECT_EQ(m.size(), 10u);
    for (const std::string &key :
         {"similarity", "quality", "freshness", "popularity", "trending", "creator_affinity",
          "exploration", "duration_match", "repetition_penalty", "impression_penalty"}) {
        EXPECT_EQ(m.count(key), 1u) << "missing key " << key;
    }
}

TEST(WeightedRankerTest, PenaltiesStoredNegativeAndSubtracted) {
    // Candidate creator 5, topic 3. History (PartialWatch, not duration-positive so duration stays
    // neutral): 2 of 4 events share creator/topic => repetition 0.5. impressionCount at the log
    // scale => impression feature 1.0.
    //  similarity: (0+1)/2=0.5      -> 0.50*0.5 = 0.25
    //  quality:    0.0              -> 0
    //  freshness:  createdAt==now=>1-> 0.08
    //  popularity: 0.5 (degenerate) -> 0.035
    //  trending:   0                -> 0
    //  affinity:   0                -> 0
    //  exploration:0                -> 0
    //  duration:   neutral 0.5      -> 0.025
    //  repetition: 0.5              -> -0.15*0.5 = -0.075
    //  impression: 1.0              -> -0.05*1.0 = -0.05
    //  score = 0.265
    std::vector<rr::Reel> reels{makeReel(0, 5, 3), makeReel(1, 9, 3), makeReel(2, 9, 8),
                                makeReel(3, 9, 8)};
    reels[0].intrinsicQuality = 0.0f;
    reels[0].impressionCount = static_cast<uint64_t>(rr::FeatureExtractor::kImpressionLogScale);
    rr::User user{};
    user.recentInteractions.push_back(event(9, 5, rr::InteractionType::PartialWatch));
    user.recentInteractions.push_back(event(1, 9, rr::InteractionType::PartialWatch));
    user.recentInteractions.push_back(event(2, 9, rr::InteractionType::PartialWatch));
    user.recentInteractions.push_back(event(3, 9, rr::InteractionType::PartialWatch));

    rr::WeightedRanker ranker(reels, {});
    const std::vector<rr::Candidate> ranked =
        ranker.rank(user, std::vector<rr::Candidate>{cand(0, 0.0f)}, 0);
    ASSERT_EQ(ranked.size(), 1u);
    const auto &m = ranked.front().featureContributions;
    EXPECT_NEAR(m.at("repetition_penalty"), -0.075f, 1e-5);
    EXPECT_NEAR(m.at("impression_penalty"), -0.05f, 1e-5);
    EXPECT_NEAR(ranked.front().rankingScore, 0.265f, 1e-5);
}

TEST(WeightedRankerTest, WeightConfigPlumbing) {
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};

    rr::RankingConfig doubled{};
    doubled.similarityWeight = 1.0; // was 0.50
    rr::WeightedRanker base(reels, {});
    rr::WeightedRanker scaled(reels, doubled);

    const rr::Candidate b = base.rank(user, std::vector<rr::Candidate>{cand(0, 0.6f)}, 0).front();
    const rr::Candidate s = scaled.rank(user, std::vector<rr::Candidate>{cand(0, 0.6f)}, 0).front();

    // similarity feature is 0.8; contribution doubles from 0.40 to 0.80, score shifts by +0.40.
    EXPECT_NEAR(b.featureContributions.at("similarity"), 0.40f, 1e-5);
    EXPECT_NEAR(s.featureContributions.at("similarity"), 0.80f, 1e-5);
    EXPECT_NEAR(s.rankingScore - b.rankingScore, 0.40f, 1e-5);
}

TEST(WeightedRankerTest, SortsDescendingByScore) {
    // Distinct similarities => distinct scores => strictly descending output.
    std::vector<rr::Reel> reels{makeReel(0), makeReel(1), makeReel(2)};
    rr::User user{};
    rr::WeightedRanker ranker(reels, {});
    std::vector<rr::Candidate> pool{cand(0, 0.1f), cand(1, 0.9f), cand(2, 0.5f)};
    const std::vector<rr::Candidate> ranked = ranker.rank(user, pool, 0);
    ASSERT_EQ(ranked.size(), 3u);
    EXPECT_EQ(ranked[0].reelId.value, 1u); // highest similarity
    EXPECT_EQ(ranked[1].reelId.value, 2u);
    EXPECT_EQ(ranked[2].reelId.value, 0u);
    EXPECT_GE(ranked[0].rankingScore, ranked[1].rankingScore);
    EXPECT_GE(ranked[1].rankingScore, ranked[2].rankingScore);
}

TEST(WeightedRankerTest, TieBreaksByAscendingReelId) {
    // Two identical reels => identical scores => ascending ReelId (2 before 5).
    std::vector<rr::Reel> reels(6);
    for (uint32_t i = 0; i < 6; ++i) {
        reels[i] = makeReel(i);
    }
    rr::User user{};
    rr::WeightedRanker ranker(reels, {});
    std::vector<rr::Candidate> pool{cand(5, 0.5f), cand(2, 0.5f)};
    const std::vector<rr::Candidate> ranked = ranker.rank(user, pool, 0);
    ASSERT_EQ(ranked.size(), 2u);
    EXPECT_FLOAT_EQ(ranked[0].rankingScore, ranked[1].rankingScore);
    EXPECT_EQ(ranked[0].reelId.value, 2u);
    EXPECT_EQ(ranked[1].reelId.value, 5u);
}

TEST(WeightedRankerTest, OutputIsPermutationOfInput) {
    std::vector<rr::Reel> reels{makeReel(0), makeReel(1), makeReel(2), makeReel(3)};
    rr::User user{};
    rr::WeightedRanker ranker(reels, {});
    std::vector<rr::Candidate> pool{cand(3, 0.2f), cand(1, 0.7f), cand(0, -0.3f), cand(2, 0.5f)};
    const std::vector<rr::Candidate> ranked = ranker.rank(user, pool, 0);
    EXPECT_EQ(ranked.size(), pool.size());
    for (const rr::Candidate &in : pool) {
        (void)findById(ranked, in.reelId.value); // each input id appears exactly once
    }
}
