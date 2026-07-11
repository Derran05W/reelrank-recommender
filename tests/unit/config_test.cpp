#include "rr/infrastructure/config.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

using json = nlohmann::json;
using namespace rr;

TEST(ConfigTest, DefaultEqualsEmptyParse) {
    ExperimentConfig def;
    auto parsed = json::parse("{}").get<ExperimentConfig>();
    EXPECT_EQ(def, parsed);
}

TEST(ConfigTest, SimulationOverrides) {
    json j = {{"simulation",
               {{"seed", 7},
                {"users", 5},
                {"reels", 9},
                {"creators", 3},
                {"topics", 8},
                {"dimensions", 128},
                {"interactions_per_user", 11}}}};
    auto c = j.get<ExperimentConfig>();
    EXPECT_EQ(c.simulation.seed, 7u);
    EXPECT_EQ(c.simulation.users, 5u);
    EXPECT_EQ(c.simulation.reels, 9u);
    EXPECT_EQ(c.simulation.creators, 3u);
    EXPECT_EQ(c.simulation.topics, 8u);
    EXPECT_EQ(c.simulation.dimensions, 128u);
    EXPECT_EQ(c.simulation.interactionsPerUser, 11u);
}

TEST(ConfigTest, AllBlocksParse) {
    json j = {
        {"recommendation",
         {{"feed_size", 20},
          {"vector_candidates", 111},
          {"popular_candidates", 22},
          {"fresh_candidates", 33},
          {"exploration_candidates", 44}}},
        {"hnsw", {{"m", 32}, {"ef_construction", 300}, {"ef_search", 128}}},
        {"ranking",
         {{"similarity_weight", 0.4},
          {"quality_weight", 0.2},
          {"freshness_weight", 0.1},
          {"popularity_weight", 0.05},
          {"trending_weight", 0.1},
          {"creator_affinity_weight", 0.1},
          {"exploration_weight", 0.05},
          {"repetition_penalty", 0.25}}},
        {"learning",
         {{"long_term_rate", 0.03},
          {"session_rate", 0.2},
          {"recent_window", 30},
          {"long_term_weight", 0.7},
          {"session_weight", 0.3}}},
        {"exploration", {{"enabled", false}, {"epsilon", 0.1}}},
        {"diversity",
         {{"enabled", false}, {"max_per_creator", 3}, {"max_per_topic", 5}, {"mmr_lambda", 0.5}}},
    };
    auto c = j.get<ExperimentConfig>();
    EXPECT_EQ(c.recommendation.feedSize, 20u);
    EXPECT_EQ(c.recommendation.vectorCandidates, 111u);
    EXPECT_EQ(c.hnsw.m, 32u);
    EXPECT_EQ(c.hnsw.efConstruction, 300u);
    EXPECT_EQ(c.hnsw.efSearch, 128u);
    EXPECT_DOUBLE_EQ(c.ranking.similarityWeight, 0.4);
    EXPECT_DOUBLE_EQ(c.ranking.repetitionPenalty, 0.25);
    EXPECT_DOUBLE_EQ(c.learning.longTermRate, 0.03);
    EXPECT_EQ(c.learning.recentWindow, 30u);
    EXPECT_DOUBLE_EQ(c.learning.longTermWeight, 0.7);
    EXPECT_DOUBLE_EQ(c.learning.sessionWeight, 0.3);
    EXPECT_FALSE(c.exploration.enabled);
    EXPECT_DOUBLE_EQ(c.exploration.epsilon, 0.1);
    EXPECT_FALSE(c.diversity.enabled);
    EXPECT_EQ(c.diversity.maxPerCreator, 3u);
    EXPECT_DOUBLE_EQ(c.diversity.mmrLambda, 0.5);
}

TEST(ConfigTest, UnknownTopLevelKeyThrows) {
    json j = {{"bogus", 1}};
    try {
        j.get<ExperimentConfig>();
        FAIL() << "expected throw";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find("bogus"), std::string::npos);
    }
}

TEST(ConfigTest, UnknownNestedKeyThrows) {
    json j = {{"simulation", {{"seed", 1}, {"typo_key", 2}}}};
    try {
        j.get<ExperimentConfig>();
        FAIL() << "expected throw";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("typo_key"), std::string::npos);
        EXPECT_NE(msg.find("simulation"), std::string::npos);
    }
}

TEST(ConfigTest, RoundTrip) {
    ExperimentConfig c;
    c.simulation.seed = 99;
    c.simulation.users = 123;
    c.recommendation.feedSize = 25;
    c.hnsw.efSearch = 77;
    c.ranking.similarityWeight = 0.42;
    c.learning.sessionWeight = 0.31;
    c.exploration.epsilon = 0.09;
    c.diversity.mmrLambda = 0.66;
    c.behaviour.alpha = 3.3;
    c.behaviour.skipBias = 0.8;
    c.reward.shareWeight = 0.25;
    c.algorithm = RecommendationAlgorithm::HnswRankerDiversity;
    json j = c;
    auto back = j.get<ExperimentConfig>();
    EXPECT_EQ(c, back);
}

TEST(ConfigTest, UnknownBehaviourKeyThrows) {
    json j = {{"behaviour", {{"alpha", 2.0}, {"alhpa", 2.0}}}};
    try {
        j.get<ExperimentConfig>();
        FAIL() << "expected throw";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("alhpa"), std::string::npos);
        EXPECT_NE(msg.find("behaviour"), std::string::npos);
    }
}

TEST(ConfigTest, UnknownRewardKeyThrows) {
    json j = {{"reward", {{"like_weight", 0.2}, {"likeweight", 0.2}}}};
    try {
        j.get<ExperimentConfig>();
        FAIL() << "expected throw";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("likeweight"), std::string::npos);
        EXPECT_NE(msg.find("reward"), std::string::npos);
    }
}

// The reward weights default to the TDD 10.5 suggested values exactly. (BehaviourConfig defaults
// are deliberately NOT pinned here — the TDD suggests none, so they are calibration surface for
// the behaviour model's statistical tests.)
TEST(ConfigTest, RewardDefaultsMatchTdd) {
    RewardConfig r;
    EXPECT_DOUBLE_EQ(r.watchRatioWeight, 0.45);
    EXPECT_DOUBLE_EQ(r.watchSecondsWeight, 0.15);
    EXPECT_DOUBLE_EQ(r.likeWeight, 0.15);
    EXPECT_DOUBLE_EQ(r.shareWeight, 0.20);
    EXPECT_DOUBLE_EQ(r.followWeight, 0.15);
    EXPECT_DOUBLE_EQ(r.instantSkipPenalty, 0.35);
    EXPECT_DOUBLE_EQ(r.notInterestedPenalty, 0.75);
}

TEST(ConfigTest, AlgorithmEnumRoundTrip) {
    const RecommendationAlgorithm all[] = {
        RecommendationAlgorithm::Random,
        RecommendationAlgorithm::Popularity,
        RecommendationAlgorithm::ExactVector,
        RecommendationAlgorithm::Hnsw,
        RecommendationAlgorithm::HnswRanker,
        RecommendationAlgorithm::HnswRankerDiversity,
        RecommendationAlgorithm::HnswRankerExploration,
    };
    for (auto a : all) {
        json j = a;
        EXPECT_EQ(j.get<RecommendationAlgorithm>(), a);
    }
    EXPECT_EQ(json("hnsw_ranker").get<RecommendationAlgorithm>(),
              RecommendationAlgorithm::HnswRanker);
    EXPECT_THROW(json("not_an_algo").get<RecommendationAlgorithm>(), std::invalid_argument);
}

TEST(ConfigTest, LoadSmallConfig) {
    std::filesystem::path root = RR_SOURCE_DIR;
    auto c = loadExperimentConfig(root / "configs" / "small.json");
    EXPECT_EQ(c.simulation.users, 1000u);
    EXPECT_EQ(c.simulation.reels, 10000u);
    EXPECT_EQ(c.simulation.creators, 500u);
    EXPECT_EQ(c.recommendation.vectorCandidates, 200u);
}

TEST(ConfigTest, AllShippedConfigsParse) {
    std::filesystem::path root = RR_SOURCE_DIR;
    for (const char *name : {"small", "medium", "large", "benchmark"}) {
        auto c = loadExperimentConfig(root / "configs" / (std::string(name) + ".json"));
        EXPECT_GT(c.simulation.users, 0u) << name;
    }
}

TEST(ConfigTest, MissingFileThrows) {
    try {
        loadExperimentConfig("/nonexistent/path/does_not_exist.json");
        FAIL() << "expected throw";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find("does_not_exist.json"), std::string::npos);
    }
}
