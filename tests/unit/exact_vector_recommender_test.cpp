#include "rr/recommendation/exact_vector_recommender.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/recommender_factory.hpp"

namespace {

rr::Reel makeReel(uint32_t id, rr::Embedding embedding, bool active = true) {
    rr::normalize(embedding);
    rr::Reel reel{};
    reel.id = rr::ReelId{id};
    reel.durationSeconds = 10.0f;
    reel.active = active;
    reel.embedding = std::move(embedding);
    return reel;
}

rr::User makeUser(rr::Embedding preference, std::vector<uint32_t> seen = {}) {
    rr::normalize(preference);
    rr::User user{};
    user.id = rr::UserId{0};
    user.estimatedPreference = std::move(preference);
    for (uint32_t s : seen) {
        user.seenReels.insert(rr::ReelId{s});
    }
    return user;
}

rr::ExperimentConfig config2d() {
    rr::ExperimentConfig config{};
    config.simulation.dimensions = 2;
    return config;
}

rr::RecommendationRequest request(std::size_t feedSize) {
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{0};
    req.feedSize = feedSize;
    return req;
}

std::vector<uint32_t> feedIds(const rr::RecommendationResponse &response) {
    std::vector<uint32_t> ids;
    for (const rr::RankedReel &r : response.reels) {
        ids.push_back(r.reelId.value);
    }
    return ids;
}

// Reels arranged around the query direction {1,0}. Descending cosine order is 0, 2, 1, 3.
std::vector<rr::Reel> orientedReels() {
    return {makeReel(0, {1.0f, 0.0f}), makeReel(1, {0.0f, 1.0f}), makeReel(2, {1.0f, 1.0f}),
            makeReel(3, {-1.0f, 0.0f})};
}

} // namespace

TEST(ExactVectorRecommenderTest, OrdersByDescendingCosineWithScoresAndRanks) {
    const std::vector<rr::Reel> reels = orientedReels();
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    const rr::RecommendationResponse response = rec.recommend(request(4));

    ASSERT_EQ(response.reels.size(), 4u);
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{0, 2, 1, 3}));
    for (std::size_t i = 0; i < response.reels.size(); ++i) {
        EXPECT_EQ(response.reels[i].rank, i);
        ASSERT_EQ(response.reels[i].sources.size(), 1u);
        EXPECT_EQ(response.reels[i].sources[0], rr::CandidateSource::VectorExact);
    }
    // score == cosine similarity to the query.
    EXPECT_NEAR(response.reels[0].score, 1.0f, 1e-5f);     // {1,0}
    EXPECT_NEAR(response.reels[1].score, 0.70710f, 1e-4f); // {1,1} normalized
    EXPECT_NEAR(response.reels[2].score, 0.0f, 1e-5f);     // {0,1}
    EXPECT_NEAR(response.reels[3].score, -1.0f, 1e-5f);    // {-1,0}
    // Scores must be non-increasing across the feed.
    for (std::size_t i = 1; i < response.reels.size(); ++i) {
        EXPECT_GE(response.reels[i - 1].score, response.reels[i].score);
    }
}

TEST(ExactVectorRecommenderTest, ExcludesSeenReels) {
    const std::vector<rr::Reel> reels = orientedReels();
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f}, /*seen=*/{0})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    const rr::RecommendationResponse response = rec.recommend(request(3));
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{2, 1, 3})); // reel 0 filtered out
}

TEST(ExactVectorRecommenderTest, ExcludesInactiveReelsFromIndex) {
    std::vector<rr::Reel> reels = orientedReels();
    reels[3].active = false; // deactivate the {-1,0} reel before construction -> never indexed
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    const rr::RecommendationResponse response = rec.recommend(request(10));
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{0, 2, 1}));
}

TEST(ExactVectorRecommenderTest, ReturnsExactFeedSizeWhenPoolLargerThanFeed) {
    const std::vector<rr::Reel> reels = orientedReels();
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    const rr::RecommendationResponse response = rec.recommend(request(2));
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(response.candidatesRanked, 2u);
}

TEST(ExactVectorRecommenderTest, DeterministicAcrossInstances) {
    const std::vector<rr::Reel> reels = orientedReels();
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender a(deps, rr::Rng(11));
    rr::ExactVectorRecommender b(deps, rr::Rng(999));
    EXPECT_EQ(feedIds(a.recommend(request(4))), feedIds(b.recommend(request(4))));
}

TEST(ExactVectorRecommenderTest, FallsBackToFullScanWhenTopKAllIneligible) {
    // Reels 0 and 1 sit nearest the query, reel 2 is far. After indexing, deactivate 0 and 1 so
    // the over-fetched top-k window (which is {0,1}) yields nothing eligible, forcing the
    // full-index fallback to still surface reel 2.
    std::vector<rr::Reel> reels{makeReel(0, {1.0f, 0.0f}), makeReel(1, {0.9f, 0.1f}),
                                makeReel(2, {0.0f, 1.0f})};
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    reels[0].active = false; // mutate live state after the index is built
    reels[1].active = false;

    const rr::RecommendationResponse response = rec.recommend(request(2));
    ASSERT_EQ(response.reels.size(), 1u); // only reel 2 remains eligible
    EXPECT_EQ(response.reels[0].reelId, rr::ReelId{2});
}

TEST(ExactVectorRecommenderTest, NameIsExactVector) {
    const std::vector<rr::Reel> reels = orientedReels();
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};
    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    EXPECT_EQ(rec.name(), "exact_vector");
}

// The recommender populates all four latency fields (Phase 5 task 3). reranking is always 0.0
// (no reranking stage); the others are non-negative and total encloses retrieval + ranking.
TEST(ExactVectorRecommenderTest, PopulatesPerStageLatencies) {
    const std::vector<rr::Reel> reels = orientedReels();
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    const rr::RecommendationResponse response = rec.recommend(request(4));

    EXPECT_GE(response.retrievalLatencyMs, 0.0);
    EXPECT_GE(response.rankingLatencyMs, 0.0);
    EXPECT_DOUBLE_EQ(response.rerankingLatencyMs, 0.0);
    EXPECT_GE(response.totalLatencyMs, response.retrievalLatencyMs);
    EXPECT_GE(response.totalLatencyMs, response.rankingLatencyMs);
    EXPECT_EQ(response.candidatesRetrieved, 4u);
    EXPECT_EQ(response.candidatesRanked, response.reels.size());
}

// The evaluation hook exposes the recommender's own exact index (TDD 18.1), sized to the active
// reels. Random/Popularity return nullptr; this recommender must not.
TEST(ExactVectorRecommenderTest, RetrievalIndexExposesActiveReelIndex) {
    std::vector<rr::Reel> reels = orientedReels();
    reels[3].active = false; // one inactive reel -> excluded from the index
    const std::vector<rr::User> users{makeUser({1.0f, 0.0f})};
    const rr::ExperimentConfig config = config2d();
    const rr::RecommenderDeps deps{reels, users, config};

    rr::ExactVectorRecommender rec(deps, rr::Rng(0));
    const rr::VectorIndex *index = rec.retrievalIndex();
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->size(), 3u); // 3 active reels indexed
}
