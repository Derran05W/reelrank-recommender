// Phase 6 integration tests: the HNSW+ranker pipeline end-to-end (TDD 16.5), ranking
// explanations well-formed on every feed item (TDD 14.4), full-run determinism, and the
// statistical exit criterion — the ranker beats the raw-similarity HNSW recommender (TDD 3,
// "how much does second-stage ranking improve the feed").
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/evaluation/experiment_runner.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/simulation/dataset_generator.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

// The eleven contribution keys the WeightedRanker must emit on every ranked candidate (frozen in
// weighted_ranker.hpp; penalties are stored as negative values). "session_topic" joined in
// Phase 7 with the session-topic-similarity feature.
const char *const kContributionKeys[] = {
    "similarity",       "quality",     "freshness",      "popularity",         "trending",
    "creator_affinity", "exploration", "duration_match", "repetition_penalty", "impression_penalty",
    "session_topic"};

std::string readFile(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Same-shape config as baseline_comparison_test.cpp: small but statistically meaningful, same
// seed both arms so only the recommender differs.
ExperimentConfig comparisonConfig(RecommendationAlgorithm algo) {
    ExperimentConfig c;
    c.simulation.seed = 20260711;
    c.simulation.users = 300;
    c.simulation.reels = 3000;
    c.simulation.creators = 150;
    c.simulation.topics = 16;
    c.simulation.dimensions = 32;
    c.simulation.interactionsPerUser = 30;
    c.recommendation.feedSize = 10;
    c.recommendation.vectorCandidates = 300;
    c.evaluation.oracleSampleRate = 0.05;
    c.algorithm = algo;
    return c;
}

MetricsSummary runArm(RecommendationAlgorithm algo, const fs::path &root) {
    fs::remove_all(root);
    ExperimentRunner runner(comparisonConfig(algo), root);
    return runner.run().overall;
}

} // namespace

// End-to-end: a cold hnsw_ranker feed is well-formed and every item carries a complete, additive
// explanation (TDD 14.4). Cold users exercise the neutral/absent paths of the personalized
// features (empty creatorAffinity, no recent interactions) — the keys must be present anyway.
TEST(RankerPipelineTest, EndToEndFeedWellFormedWithExplanations) {
    ExperimentConfig config{};
    config.simulation.reels = 500;
    config.simulation.users = 20;
    config.simulation.creators = 50;
    config.simulation.topics = 8;
    config.simulation.dimensions = 16;
    config.algorithm = RecommendationAlgorithm::HnswRanker;

    GeneratedDataset ds = generateDataset(config.simulation, 7u);
    for (std::size_t i = 0; i < ds.users.size(); ++i) {
        ds.users[i].estimatedPreference = ds.hiddenStates[i].hiddenPreference;
    }

    RecommenderDeps deps{ds.reels, ds.users, config};
    auto rec =
        makeRecommender(RecommendationAlgorithm::HnswRanker, deps, forkRng(7u, "recommender"));

    for (const User &user : ds.users) {
        RecommendationRequest req{};
        req.userId = user.id;
        req.feedSize = 10;
        req.candidateLimit = 100;

        const RecommendationResponse response = rec->recommend(req);

        ASSERT_FALSE(response.reels.empty());
        EXPECT_LE(response.reels.size(), req.feedSize);
        EXPECT_LE(response.candidatesRanked, req.candidateLimit);
        // Four sources feed the merge, so the PRE-dedup retrieved count may exceed the pool cap;
        // it must still be at least the ranked count.
        EXPECT_GE(response.candidatesRetrieved, response.candidatesRanked);
        EXPECT_GE(response.totalLatencyMs, 0.0);
        EXPECT_GE(response.rankingLatencyMs, 0.0);

        std::unordered_set<uint32_t> ids;
        float prevScore = std::numeric_limits<float>::infinity();
        for (const RankedReel &item : response.reels) {
            EXPECT_TRUE(ids.insert(item.reelId.value).second) << "duplicate reel in feed";
            EXPECT_FALSE(item.sources.empty()) << "source labels lost on the ranked path";

            // Ranked order is non-increasing in score (pre-diversity, phase-6 property).
            EXPECT_LE(item.score, prevScore);
            prevScore = item.score;

            // Explanation completeness + additivity: all eleven keys, sum == score.
            EXPECT_EQ(item.featureContributions.size(), std::size(kContributionKeys));
            float sum = 0.0f;
            for (const char *key : kContributionKeys) {
                auto it = item.featureContributions.find(key);
                ASSERT_NE(it, item.featureContributions.end()) << "missing contribution " << key;
                sum += it->second;
            }
            EXPECT_NEAR(sum, item.score, 1e-4f);
        }
    }
}

// Full-run determinism for the ranker pipeline: same config + seed twice => byte-identical
// deterministic outputs (latency/metadata excluded — wall clock, TDD 24.6).
TEST(RankerPipelineTest, FullRunDeterminismByteIdentical) {
    ExperimentConfig c;
    c.simulation.seed = 99;
    c.simulation.users = 30;
    c.simulation.reels = 300;
    c.simulation.creators = 15;
    c.simulation.topics = 8;
    c.simulation.dimensions = 16;
    c.simulation.interactionsPerUser = 10;
    c.recommendation.feedSize = 5;
    c.recommendation.vectorCandidates = 100;
    c.evaluation.oracleSampleRate = 0.5;
    c.algorithm = RecommendationAlgorithm::HnswRanker;

    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_ranker_det_a";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_ranker_det_b";
    fs::remove_all(rootA);
    fs::remove_all(rootB);
    ExperimentRunner runnerA(c, rootA);
    ExperimentRunner runnerB(c, rootB);
    ExperimentResult a = runnerA.run();
    ExperimentResult b = runnerB.run();

    for (const char *name : {"config.json", "retrieval_metrics.csv", "recommendation_metrics.csv",
                             "learning_curve.csv", "regret_curve.csv"}) {
        EXPECT_EQ(readFile(a.directory / name), readFile(b.directory / name))
            << name << " differs between two same-seed hnsw_ranker runs";
    }
}

// The phase-6 statistical exit criterion (plan task 5 / TDD 3): second-stage ranking must beat
// the raw-similarity HNSW recommender on REWARD per impression — reward (TDD 10.5) is the
// composite objective the ranker optimizes for (watch + likes + shares). Mean true affinity is
// reported for the record: the ranker deliberately trades some similarity-driven affinity for
// quality/popularity signal that converts to engagement, so affinity may dip while reward rises
// (measured and published in results/published/phase6/).
TEST(RankerPipelineTest, RankerBeatsRawSimilarityHnswOnReward) {
    const MetricsSummary hnsw =
        runArm(RecommendationAlgorithm::Hnsw, fs::path(::testing::TempDir()) / "rr_cmp_hnsw");
    const MetricsSummary ranker = runArm(RecommendationAlgorithm::HnswRanker,
                                         fs::path(::testing::TempDir()) / "rr_cmp_hnsw_ranker");

    std::cout << "[ metrics  ] reward/impression: ranker " << ranker.rewardPerImpression
              << " vs hnsw " << hnsw.rewardPerImpression << "\n"
              << "[ metrics  ] mean true affinity: ranker " << ranker.meanTrueAffinity
              << " vs hnsw " << hnsw.meanTrueAffinity << "\n";

    EXPECT_GT(ranker.rewardPerImpression, hnsw.rewardPerImpression)
        << "ranker reward " << ranker.rewardPerImpression << " vs hnsw "
        << hnsw.rewardPerImpression;
}
