#include "rr/recommendation/recommender_factory.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/vector_index.hpp"

namespace {

rr::Reel makeReel(uint32_t id, rr::Embedding embedding) {
    rr::normalize(embedding);
    rr::Reel reel{};
    reel.id = rr::ReelId{id};
    reel.durationSeconds = 10.0f;
    reel.active = true;
    reel.embedding = std::move(embedding);
    return reel;
}

rr::User makeUser(uint32_t id, rr::Embedding preference) {
    rr::normalize(preference);
    rr::User user{};
    user.id = rr::UserId{id};
    user.estimatedPreference = std::move(preference);
    return user;
}

// A small dense dataset valid for all three baselines (ExactVector builds a 2-D index over it).
struct Fixture {
    std::vector<rr::Reel> reels{makeReel(0, {1.0f, 0.0f}), makeReel(1, {0.0f, 1.0f}),
                                makeReel(2, {1.0f, 1.0f})};
    std::vector<rr::User> users{makeUser(0, {1.0f, 0.0f})};
    rr::ExperimentConfig config = [] {
        rr::ExperimentConfig c{};
        c.simulation.dimensions = 2;
        return c;
    }();
    rr::RecommenderDeps deps() const { return rr::RecommenderDeps{reels, users, config}; }
};

} // namespace

TEST(RecommenderFactoryTest, NamesMatchToStringForImplementedAlgorithms) {
    Fixture fx;
    // Hnsw joins the baselines in Phase 5.
    for (const rr::RecommendationAlgorithm algo :
         {rr::RecommendationAlgorithm::Random, rr::RecommendationAlgorithm::Popularity,
          rr::RecommendationAlgorithm::ExactVector, rr::RecommendationAlgorithm::Hnsw}) {
        auto rec = rr::makeRecommender(algo, fx.deps(), rr::Rng(1));
        ASSERT_NE(rec, nullptr);
        EXPECT_EQ(rec->name(), rr::toString(algo));
    }
}

TEST(RecommenderFactoryTest, HnswExposesRetrievalIndex) {
    Fixture fx;
    auto rec = rr::makeRecommender(rr::RecommendationAlgorithm::Hnsw, fx.deps(), rr::Rng(1));
    ASSERT_NE(rec, nullptr);
    ASSERT_NE(rec->retrievalIndex(), nullptr); // evaluation hook wired (TDD 18.1)
    EXPECT_EQ(rec->retrievalIndex()->size(), fx.reels.size());
}

TEST(RecommenderFactoryTest, UnimplementedAlgorithmsThrowInvalidArgument) {
    Fixture fx;
    // The ranker/diversity/exploration variants still arrive in Phases 6/8/9.
    for (const rr::RecommendationAlgorithm algo :
         {rr::RecommendationAlgorithm::HnswRanker, rr::RecommendationAlgorithm::HnswRankerDiversity,
          rr::RecommendationAlgorithm::HnswRankerExploration}) {
        EXPECT_THROW(rr::makeRecommender(algo, fx.deps(), rr::Rng(1)), std::invalid_argument);
    }
}

TEST(RecommenderFactoryTest, NonDenseReelIdsThrow) {
    Fixture fx;
    fx.reels[1].id = rr::ReelId{99}; // break the dense-id invariant
    EXPECT_THROW(rr::makeRecommender(rr::RecommendationAlgorithm::Random, fx.deps(), rr::Rng(1)),
                 std::invalid_argument);
}

TEST(RecommenderFactoryTest, NonDenseUserIdsThrow) {
    Fixture fx;
    fx.users[0].id = rr::UserId{5}; // break the dense-id invariant
    EXPECT_THROW(
        rr::makeRecommender(rr::RecommendationAlgorithm::Popularity, fx.deps(), rr::Rng(1)),
        std::invalid_argument);
}

TEST(RecommenderFactoryTest, ProducesWorkingRecommender) {
    Fixture fx;
    auto rec = rr::makeRecommender(rr::RecommendationAlgorithm::ExactVector, fx.deps(), rr::Rng(1));
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{0};
    req.feedSize = 2;
    const rr::RecommendationResponse response = rec->recommend(req);
    EXPECT_EQ(response.reels.size(), 2u);
    EXPECT_EQ(response.reels[0].reelId, rr::ReelId{0}); // {1,0} is nearest the {1,0} query
}
