// Property tests for the Phase 5 pipeline (TDD 24.3 seed sweep). Over many seeds and a small
// generated dataset, the HNSW recommender's feeds must respect the request bounds and never
// contain an inactive, seen, or duplicate reel — regardless of the graph the seed produces.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/simulation/dataset_generator.hpp"

namespace {

rr::ExperimentConfig smallConfig() {
    rr::ExperimentConfig config{};
    config.simulation.reels = 500;
    config.simulation.users = 20;
    config.simulation.creators = 50;
    config.simulation.topics = 8;
    config.simulation.dimensions = 16;
    config.algorithm = rr::RecommendationAlgorithm::Hnsw;
    return config;
}

class PipelinePropertyTest : public ::testing::TestWithParam<uint64_t> {};

} // namespace

TEST_P(PipelinePropertyTest, HnswFeedsRespectBoundsAndEligibility) {
    const uint64_t seed = GetParam();
    const rr::ExperimentConfig config = smallConfig();
    rr::GeneratedDataset ds = rr::generateDataset(config.simulation, seed);
    // Generated users have an empty estimatedPreference (cold-start init is a later phase); give
    // each a distinct query equal to their normalized hidden preference (test-side setup only).
    ASSERT_EQ(ds.users.size(), ds.hiddenStates.size());
    for (std::size_t i = 0; i < ds.users.size(); ++i) {
        ds.users[i].estimatedPreference = ds.hiddenStates[i].hiddenPreference;
    }

    rr::RecommenderDeps deps{ds.reels, ds.users, config};
    auto rec = rr::makeRecommender(rr::RecommendationAlgorithm::Hnsw, deps,
                                   rr::forkRng(seed, "recommender"));

    // Inject filter pressure AFTER the index is built: deactivate some indexed reels and mark
    // some indexed reels as seen for every user. The orchestrator reads live state, so these must
    // be filtered out of every feed.
    for (std::size_t i = 0; i < 5 && i < ds.reels.size(); ++i) {
        ds.reels[i].active = false;
    }
    for (rr::User &user : ds.users) {
        for (uint32_t s = 10; s < 15 && s < ds.reels.size(); ++s) {
            user.seenReels.insert(rr::ReelId{s});
        }
    }

    const std::size_t feedSize = 10;
    const std::size_t candidateLimit = 100;
    for (const rr::User &user : ds.users) {
        rr::RecommendationRequest req{};
        req.userId = user.id;
        req.feedSize = feedSize;
        req.candidateLimit = candidateLimit;

        const rr::RecommendationResponse response = rec->recommend(req);

        // Bounds: candidate counts never exceed the configured limits; feed never exceeds feed
        // size; and each stage count is monotone (retrieved >= ranked >= feed).
        EXPECT_LE(response.reels.size(), feedSize);
        EXPECT_LE(response.candidatesRetrieved, candidateLimit);
        EXPECT_LE(response.candidatesRanked, candidateLimit);
        EXPECT_LE(response.candidatesRanked, response.candidatesRetrieved);
        EXPECT_LE(response.reels.size(), response.candidatesRanked);

        // Eligibility: no inactive, no seen, no duplicate reels.
        std::unordered_set<uint32_t> ids;
        for (const rr::RankedReel &r : response.reels) {
            ASSERT_LT(r.reelId.value, ds.reels.size());
            EXPECT_TRUE(ds.reels[r.reelId.value].active) << "inactive reel in feed";
            EXPECT_FALSE(user.seenReels.contains(r.reelId)) << "seen reel in feed";
            EXPECT_TRUE(ids.insert(r.reelId.value).second) << "duplicate reel in feed";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SeedSweep, PipelinePropertyTest,
                         ::testing::Range<uint64_t>(1, 26)); // 25 seeds
