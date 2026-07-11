// Integration tests for Phase 5: HNSW served end-to-end through the factory + orchestrator,
// compared against the exact ceiling on an identical seed/dataset (TDD 28 P5). Covers top-k
// overlap, full latency/count population, and same-seed determinism.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
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

rr::ExperimentConfig mediumishConfig() {
    rr::ExperimentConfig config{};
    config.simulation.reels = 1500;
    config.simulation.users = 50;
    config.simulation.creators = 100;
    config.simulation.topics = 16;
    config.simulation.dimensions = 32;
    return config;
}

// Generated users start with an empty estimatedPreference (cold-start init is a later phase). For
// a meaningful retrieval comparison we give each simulated user a distinct, well-learned estimate
// equal to their (already-normalized) hidden preference — test-side setup only.
void seedEstimatesFromHidden(rr::GeneratedDataset &ds) {
    ASSERT_EQ(ds.users.size(), ds.hiddenStates.size());
    for (std::size_t i = 0; i < ds.users.size(); ++i) {
        ds.users[i].estimatedPreference = ds.hiddenStates[i].hiddenPreference;
    }
}

rr::RecommendationRequest request(uint32_t userId, std::size_t feedSize,
                                  std::size_t candidateLimit) {
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{userId};
    req.feedSize = feedSize;
    req.candidateLimit = candidateLimit;
    return req;
}

std::unordered_set<uint32_t> feedSet(const rr::RecommendationResponse &response) {
    std::unordered_set<uint32_t> ids;
    for (const rr::RankedReel &r : response.reels) {
        ids.insert(r.reelId.value);
    }
    return ids;
}

} // namespace

TEST(HnswPipelineIntegrationTest, HnswTopKOverlapsExactCeiling) {
    const uint64_t seed = 20260711;
    const rr::ExperimentConfig config = mediumishConfig();
    rr::GeneratedDataset ds = rr::generateDataset(config.simulation, seed);
    seedEstimatesFromHidden(ds);
    rr::RecommenderDeps deps{ds.reels, ds.users, config};

    auto exact = rr::makeRecommender(rr::RecommendationAlgorithm::ExactVector, deps,
                                     rr::forkRng(seed, "recommender"));
    auto hnsw = rr::makeRecommender(rr::RecommendationAlgorithm::Hnsw, deps,
                                    rr::forkRng(seed, "recommender"));

    const std::size_t feedSize = 10;
    const std::size_t candidateLimit = 200;
    const std::size_t sampledUsers = 30;

    double overlapSum = 0.0;
    std::size_t counted = 0;
    for (uint32_t u = 0; u < sampledUsers && u < ds.users.size(); ++u) {
        const std::unordered_set<uint32_t> exactTop =
            feedSet(exact->recommend(request(u, feedSize, candidateLimit)));
        const std::unordered_set<uint32_t> hnswTop =
            feedSet(hnsw->recommend(request(u, feedSize, candidateLimit)));
        if (exactTop.empty()) {
            continue;
        }
        std::size_t shared = 0;
        for (uint32_t id : hnswTop) {
            if (exactTop.count(id) != 0) {
                ++shared;
            }
        }
        overlapSum += static_cast<double>(shared) / static_cast<double>(exactTop.size());
        ++counted;
    }

    ASSERT_GT(counted, 0u);
    const double meanOverlap = overlapSum / static_cast<double>(counted);
    std::cout << "[hnsw-vs-exact] mean top-" << feedSize << " overlap over " << counted
              << " users = " << meanOverlap << std::endl;
    // Topic-clustered data at small scale: HNSW recovers most of the exact top-k. Conservative
    // floor chosen to be well below the measured value so this is not flaky.
    EXPECT_GT(meanOverlap, 0.3);
}

TEST(HnswPipelineIntegrationTest, EndToEndPopulatesAllLatencyAndCountFields) {
    const uint64_t seed = 99;
    const rr::ExperimentConfig config = mediumishConfig();
    rr::GeneratedDataset ds = rr::generateDataset(config.simulation, seed);
    seedEstimatesFromHidden(ds);
    rr::RecommenderDeps deps{ds.reels, ds.users, config};
    auto hnsw = rr::makeRecommender(rr::RecommendationAlgorithm::Hnsw, deps,
                                    rr::forkRng(seed, "recommender"));

    const rr::RecommendationResponse response = hnsw->recommend(request(0, 10, 200));

    EXPECT_GT(response.retrievalLatencyMs, 0.0); // sources actually ran
    EXPECT_GE(response.rankingLatencyMs, 0.0);
    EXPECT_GE(response.rerankingLatencyMs, 0.0);
    EXPECT_GE(response.totalLatencyMs, response.retrievalLatencyMs); // total contains each stage
    EXPECT_GE(response.totalLatencyMs, response.rankingLatencyMs);
    EXPECT_GE(response.totalLatencyMs, response.rerankingLatencyMs);
    EXPECT_GT(response.candidatesRetrieved, 0u);
    EXPECT_GT(response.candidatesRanked, 0u);
    EXPECT_LE(response.candidatesRetrieved, 200u);
}

TEST(HnswPipelineIntegrationTest, IndependentSameSeedRecommendersAreDeterministic) {
    const uint64_t seed = 4242;
    const rr::ExperimentConfig config = mediumishConfig();
    rr::GeneratedDataset ds = rr::generateDataset(config.simulation, seed);
    seedEstimatesFromHidden(ds);
    rr::RecommenderDeps deps{ds.reels, ds.users, config};

    auto a = rr::makeRecommender(rr::RecommendationAlgorithm::Hnsw, deps,
                                 rr::forkRng(seed, "recommender"));
    auto b = rr::makeRecommender(rr::RecommendationAlgorithm::Hnsw, deps,
                                 rr::forkRng(seed, "recommender"));

    for (uint32_t u = 0; u < 10 && u < ds.users.size(); ++u) {
        const rr::RecommendationResponse ra = a->recommend(request(u, 10, 200));
        const rr::RecommendationResponse rb = b->recommend(request(u, 10, 200));
        ASSERT_EQ(ra.reels.size(), rb.reels.size());
        for (std::size_t i = 0; i < ra.reels.size(); ++i) {
            EXPECT_EQ(ra.reels[i].reelId, rb.reels[i].reelId);
            EXPECT_FLOAT_EQ(ra.reels[i].score, rb.reels[i].score);
            EXPECT_EQ(ra.reels[i].rank, rb.reels[i].rank);
            EXPECT_EQ(ra.reels[i].sources, rb.reels[i].sources);
        }
    }
}
