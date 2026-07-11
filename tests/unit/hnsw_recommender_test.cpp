// Unit tests for HNSWRecommender (Phase 5 task 4, TDD 16.4). Small hand-built dataset; graph
// quality is irrelevant here — these check wiring, the retrievalIndex() hook, the eligibility
// filters, and same-seed determinism.
#include "rr/recommendation/hnsw_recommender.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"

namespace {

rr::Reel makeReel(uint32_t id, rr::Embedding embedding, bool active = true) {
    rr::normalize(embedding);
    rr::Reel reel{};
    reel.id = rr::ReelId{id};
    reel.active = active;
    reel.durationSeconds = 10.0f;
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

rr::ExperimentConfig config3d() {
    rr::ExperimentConfig config{};
    config.simulation.dimensions = 3;
    return config;
}

// Ten reels around the unit sphere, dense ids 0..9.
std::vector<rr::Reel> makeReels() {
    return {makeReel(0, {1.0f, 0.0f, 0.0f}), makeReel(1, {0.9f, 0.1f, 0.0f}),
            makeReel(2, {0.8f, 0.2f, 0.1f}), makeReel(3, {0.5f, 0.5f, 0.0f}),
            makeReel(4, {0.3f, 0.7f, 0.1f}), makeReel(5, {0.0f, 1.0f, 0.0f}),
            makeReel(6, {0.0f, 0.9f, 0.2f}), makeReel(7, {0.1f, 0.0f, 1.0f}),
            makeReel(8, {0.2f, 0.1f, 0.9f}), makeReel(9, {-1.0f, 0.1f, 0.0f})};
}

rr::RecommendationRequest request(std::size_t feedSize, std::size_t candidateLimit) {
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{0};
    req.feedSize = feedSize;
    req.candidateLimit = candidateLimit;
    return req;
}

} // namespace

TEST(HnswRecommenderTest, NameMatchesToString) {
    std::vector<rr::Reel> reels = makeReels();
    std::vector<rr::User> users{makeUser({1.0f, 0.0f, 0.0f})};
    rr::ExperimentConfig config = config3d();
    rr::HNSWRecommender rec(rr::RecommenderDeps{reels, users, config}, rr::Rng(1));
    EXPECT_EQ(rec.name(), rr::toString(rr::RecommendationAlgorithm::Hnsw));
    EXPECT_EQ(rec.name(), "hnsw");
}

TEST(HnswRecommenderTest, RetrievalIndexExposesActiveReels) {
    std::vector<rr::Reel> reels = makeReels();
    reels[9].active = false; // one inactive reel is not indexed
    std::vector<rr::User> users{makeUser({1.0f, 0.0f, 0.0f})};
    rr::ExperimentConfig config = config3d();
    rr::HNSWRecommender rec(rr::RecommenderDeps{reels, users, config}, rr::Rng(1));

    const rr::VectorIndex *index = rec.retrievalIndex();
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->size(), 9u); // 10 reels, 1 inactive
}

TEST(HnswRecommenderTest, FeedIsEligibleAndBounded) {
    std::vector<rr::Reel> reels = makeReels();
    std::vector<rr::User> users{makeUser({1.0f, 0.0f, 0.0f})};
    rr::ExperimentConfig config = config3d();
    rr::HNSWRecommender rec(rr::RecommenderDeps{reels, users, config}, rr::Rng(1));

    const rr::RecommendationResponse response = rec.recommend(request(5, 10));
    EXPECT_LE(response.reels.size(), 5u);
    EXPECT_FALSE(response.reels.empty());
    std::unordered_set<uint32_t> seen;
    for (const rr::RankedReel &r : response.reels) {
        EXPECT_TRUE(reels[r.reelId.value].active);
        EXPECT_TRUE(seen.insert(r.reelId.value).second); // no duplicates
    }
    EXPECT_GT(response.retrievalLatencyMs, 0.0);
    EXPECT_GT(response.candidatesRetrieved, 0u);
}

TEST(HnswRecommenderTest, DropsSeenAndInactiveFromFeed) {
    std::vector<rr::Reel> reels = makeReels();
    reels[0].active = false;                                        // nearest reel is inactive
    std::vector<rr::User> users{makeUser({1.0f, 0.0f, 0.0f}, {1})}; // second-nearest is seen
    rr::ExperimentConfig config = config3d();
    rr::HNSWRecommender rec(rr::RecommenderDeps{reels, users, config}, rr::Rng(1));

    const rr::RecommendationResponse response = rec.recommend(request(10, 10));
    for (const rr::RankedReel &r : response.reels) {
        EXPECT_NE(r.reelId, rr::ReelId{0});
        EXPECT_NE(r.reelId, rr::ReelId{1});
    }
}

TEST(HnswRecommenderTest, SameSeedProducesIdenticalFeeds) {
    std::vector<rr::Reel> reels = makeReels();
    std::vector<rr::User> users{makeUser({1.0f, 0.0f, 0.0f})};
    rr::ExperimentConfig config = config3d();
    rr::HNSWRecommender a(rr::RecommenderDeps{reels, users, config}, rr::Rng(7));
    rr::HNSWRecommender b(rr::RecommenderDeps{reels, users, config}, rr::Rng(7));

    const rr::RecommendationResponse ra = a.recommend(request(5, 10));
    const rr::RecommendationResponse rb = b.recommend(request(5, 10));
    ASSERT_EQ(ra.reels.size(), rb.reels.size());
    for (std::size_t i = 0; i < ra.reels.size(); ++i) {
        EXPECT_EQ(ra.reels[i].reelId, rb.reels[i].reelId);
        EXPECT_FLOAT_EQ(ra.reels[i].score, rb.reels[i].score);
        EXPECT_EQ(ra.reels[i].rank, rb.reels[i].rank);
    }
}
