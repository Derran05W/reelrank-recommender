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
                {"interactions_per_user", 11},
                {"new_users", 50},
                {"new_users_at", 4},
                {"new_reels", 60},
                {"new_reels_at", 6}}}};
    auto c = j.get<ExperimentConfig>();
    EXPECT_EQ(c.simulation.seed, 7u);
    EXPECT_EQ(c.simulation.users, 5u);
    EXPECT_EQ(c.simulation.reels, 9u);
    EXPECT_EQ(c.simulation.creators, 3u);
    EXPECT_EQ(c.simulation.topics, 8u);
    EXPECT_EQ(c.simulation.dimensions, 128u);
    EXPECT_EQ(c.simulation.interactionsPerUser, 11u);
    EXPECT_EQ(c.simulation.newUsers, 50u);
    EXPECT_EQ(c.simulation.newUsersAt, 4u);
    EXPECT_EQ(c.simulation.newReels, 60u);
    EXPECT_EQ(c.simulation.newReelsAt, 6u);
}

TEST(ConfigTest, InjectionDefaultsDisabled) {
    // Phase 8 mid-simulation injection is opt-in: default counts are 0 (no injection).
    SimulationConfig def;
    EXPECT_EQ(def.newUsers, 0u);
    EXPECT_EQ(def.newReels, 0u);
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
          {"repetition_penalty", 0.25},
          {"session_topic_weight", 0.07}}},
        {"learning",
         {{"enabled", false},
          {"long_term_rate", 0.03},
          {"session_rate", 0.2},
          {"recent_window", 30},
          {"session_lambda", 0.88},
          {"long_term_weight", 0.7},
          {"session_weight", 0.3}}},
        {"exploration",
         {{"enabled", false},
          {"epsilon", 0.1},
          {"fresh_window_seconds", 100000.0},
          {"guaranteed_slots", 3}}},
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
    EXPECT_DOUBLE_EQ(c.ranking.sessionTopicWeight, 0.07);
    EXPECT_FALSE(c.learning.enabled);
    EXPECT_DOUBLE_EQ(c.learning.longTermRate, 0.03);
    EXPECT_EQ(c.learning.recentWindow, 30u);
    EXPECT_DOUBLE_EQ(c.learning.sessionLambda, 0.88);
    EXPECT_DOUBLE_EQ(c.learning.longTermWeight, 0.7);
    EXPECT_DOUBLE_EQ(c.learning.sessionWeight, 0.3);
    EXPECT_FALSE(c.exploration.enabled);
    EXPECT_DOUBLE_EQ(c.exploration.epsilon, 0.1);
    EXPECT_DOUBLE_EQ(c.exploration.freshWindowSeconds, 100000.0);
    EXPECT_EQ(c.exploration.guaranteedSlots, 3u);
    EXPECT_FALSE(c.diversity.enabled);
    EXPECT_EQ(c.diversity.maxPerCreator, 3u);
    EXPECT_DOUBLE_EQ(c.diversity.mmrLambda, 0.5);
}

TEST(ConfigTest, DriftDefaultsDisabled) {
    // Phase 10 scheduled drift is opt-in: the default config has no events (drift disabled), and
    // an empty-events parse equals the default (the pre-Phase-10 byte-identical contract hinges
    // on this default).
    ExperimentConfig def;
    EXPECT_TRUE(def.drift.events.empty());
    json j = {{"drift", {{"events", json::array()}}}};
    EXPECT_EQ(j.get<ExperimentConfig>(), def);
}

TEST(ConfigTest, DriftBlockParses) {
    json j = {
        {"drift",
         {{"events",
           {{{"at_interaction", 100},
             {"cohort_lo", 0.25},
             {"cohort_hi", 0.5},
             {"topic_mix", {{{"topic", 3}, {"weight", 0.6}}, {{"topic", 11}, {"weight", 0.4}}}}},
            {{"at_interaction", 150}, {"topic_mix", {{{"topic", 7}, {"weight", 1.0}}}}}}}}}};
    auto c = j.get<ExperimentConfig>();
    ASSERT_EQ(c.drift.events.size(), 2u);
    const DriftEvent &e0 = c.drift.events[0];
    EXPECT_EQ(e0.atInteraction, 100u);
    EXPECT_DOUBLE_EQ(e0.cohortLo, 0.25);
    EXPECT_DOUBLE_EQ(e0.cohortHi, 0.5);
    ASSERT_EQ(e0.topicMix.size(), 2u);
    EXPECT_EQ(e0.topicMix[0].topic, 3u);
    EXPECT_DOUBLE_EQ(e0.topicMix[0].weight, 0.6);
    EXPECT_EQ(e0.topicMix[1].topic, 11u);
    EXPECT_DOUBLE_EQ(e0.topicMix[1].weight, 0.4);
    // Cohort defaults cover the whole population.
    const DriftEvent &e1 = c.drift.events[1];
    EXPECT_DOUBLE_EQ(e1.cohortLo, 0.0);
    EXPECT_DOUBLE_EQ(e1.cohortHi, 1.0);
    // Round-trips through to_json/from_json unchanged.
    json out = c;
    EXPECT_EQ(out.get<ExperimentConfig>(), c);
}

TEST(ConfigTest, UnknownDriftKeysThrow) {
    for (const json &j :
         {json{{"drift", {{"bogus_key", 1}}}},
          json{{"drift", {{"events", {{{"at_interaction", 1}, {"typo_key", 2}}}}}}},
          json{
              {"drift",
               {{"events", {{{"topic_mix", {{{"topic", 1}, {"weight", 1.0}, {"oops", 3}}}}}}}}}}}) {
        EXPECT_THROW(j.get<ExperimentConfig>(), std::invalid_argument);
    }
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

TEST(ConfigTest, EvaluationBlockParsesAndRejectsUnknownKeys) {
    EXPECT_DOUBLE_EQ(ExperimentConfig{}.evaluation.oracleSampleRate, 0.05);
    EXPECT_DOUBLE_EQ(ExperimentConfig{}.evaluation.retrievalSampleRate, 0.02);

    json j = {{"evaluation", {{"oracle_sample_rate", 0.25}, {"retrieval_sample_rate", 0.1}}}};
    auto c = j.get<ExperimentConfig>();
    EXPECT_DOUBLE_EQ(c.evaluation.oracleSampleRate, 0.25);
    EXPECT_DOUBLE_EQ(c.evaluation.retrievalSampleRate, 0.1);

    json bad = {{"evaluation", {{"oracle_rate", 0.25}}}};
    try {
        bad.get<ExperimentConfig>();
        FAIL() << "expected throw";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("oracle_rate"), std::string::npos);
        EXPECT_NE(msg.find("evaluation"), std::string::npos);
    }
}

TEST(ConfigTest, LearningDefaults) {
    // Phase 7 additions: learning is on by default; the frozen-estimates experiment arm sets
    // enabled=false. Lambda default sits mid TDD 11.3's suggested 0.85-0.95 range.
    const ExperimentConfig c{};
    EXPECT_TRUE(c.learning.enabled);
    EXPECT_DOUBLE_EQ(c.learning.sessionLambda, 0.90);
    EXPECT_DOUBLE_EQ(c.ranking.sessionTopicWeight, 0.05);
}

TEST(ConfigTest, RoundTrip) {
    ExperimentConfig c;
    c.simulation.seed = 99;
    c.simulation.users = 123;
    c.recommendation.feedSize = 25;
    c.hnsw.efSearch = 77;
    c.ranking.similarityWeight = 0.42;
    c.ranking.sessionTopicWeight = 0.06;
    c.learning.enabled = false;
    c.learning.sessionLambda = 0.93;
    c.learning.sessionWeight = 0.31;
    c.exploration.epsilon = 0.09;
    c.diversity.mmrLambda = 0.66;
    c.diversity.useMmr = false;
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

// --- Realism V2 block (Phase 13, D17/D24) ---------------------------------------------------

TEST(ConfigTest, RealismDefaultsPreserveV1) {
    ExperimentConfig def;
    EXPECT_FALSE(def.realism.contentV2);
    EXPECT_EQ(def.realism.languages, 8u);
    EXPECT_EQ(def.realism.archetypes, defaultArchetypeCatalog());
    EXPECT_EQ(def.realism.archetypes.size(), 8u);
}

TEST(ConfigTest, RealismBlockParsesAndRoundTrips) {
    json j = {{"realism",
               {{"content_v2", true},
                {"languages", 3},
                {"archetypes", json::array({{{"name", "only"}, {"weight", 2.0}}})}}}};
    auto c = j.get<ExperimentConfig>();
    EXPECT_TRUE(c.realism.contentV2);
    EXPECT_EQ(c.realism.languages, 3u);
    ASSERT_EQ(c.realism.archetypes.size(), 1u);
    EXPECT_EQ(c.realism.archetypes[0].name, "only");
    EXPECT_DOUBLE_EQ(c.realism.archetypes[0].weight, 2.0);

    json out = c;
    auto back = out.get<ExperimentConfig>();
    EXPECT_EQ(c, back);
}

TEST(ConfigTest, UnknownRealismKeyThrows) {
    json j = {{"realism", {{"content_v3", true}}}};
    EXPECT_THROW(j.get<ExperimentConfig>(), std::invalid_argument);
}

TEST(ConfigTest, RealismValidationThrows) {
    // languages must be >= 1.
    json zeroLang = {{"realism", {{"languages", 0}}}};
    EXPECT_THROW(zeroLang.get<ExperimentConfig>(), std::invalid_argument);
    // The archetype catalog must not be empty.
    json emptyCatalog = {{"realism", {{"archetypes", json::array()}}}};
    EXPECT_THROW(emptyCatalog.get<ExperimentConfig>(), std::invalid_argument);
    // content_v2 + mid-simulation injection is a documented Phase 13 scope guard.
    json injection = {{"realism", {{"content_v2", true}}},
                      {"simulation", {{"new_users", 10}, {"new_users_at", 2}}}};
    EXPECT_THROW(injection.get<ExperimentConfig>(), std::invalid_argument);
}

// --- Realism V2 Phase 14: latent-reactions gate + behaviour_v2 block ------------------------

TEST(ConfigTest, LatentReactionsGateRequiresContentV2) {
    // Default off; parses fine alongside content_v2.
    ExperimentConfig def;
    EXPECT_FALSE(def.realism.latentReactions);
    json ok = {{"realism", {{"content_v2", true}, {"latent_reactions", true}}}};
    EXPECT_TRUE(ok.get<ExperimentConfig>().realism.latentReactions);
    // D17 gate dependency: latent_reactions without content_v2 fails at load.
    json bad = {{"realism", {{"latent_reactions", true}}}};
    EXPECT_THROW(bad.get<ExperimentConfig>(), std::invalid_argument);
}

// --- Realism V2 Phase 15: ranking V2 weights, modality rate, oracle algorithm ----------------

TEST(ConfigTest, RankingV2WeightsDefaultZeroAndParse) {
    ExperimentConfig def;
    // All V2 feature weights default 0.0 so gate-on-with-defaults ranks exactly like V1 (the
    // single-variable experiment-arm contract, Phase 15).
    EXPECT_DOUBLE_EQ(def.ranking.visualMatchWeight, 0.0);
    EXPECT_DOUBLE_EQ(def.ranking.clickbaitWeight, 0.0);
    EXPECT_DOUBLE_EQ(def.ranking.savePopularityWeight, 0.0);
    EXPECT_DOUBLE_EQ(def.learning.modalityRate, 0.02);

    json j = {{"ranking", {{"clickbait_weight", 0.2}, {"music_match_weight", 0.1}}},
              {"learning", {{"modality_rate", 0.05}}}};
    auto c = j.get<ExperimentConfig>();
    EXPECT_DOUBLE_EQ(c.ranking.clickbaitWeight, 0.2);
    EXPECT_DOUBLE_EQ(c.ranking.musicMatchWeight, 0.1);
    EXPECT_DOUBLE_EQ(c.learning.modalityRate, 0.05);
    json out = c;
    EXPECT_EQ(out.get<ExperimentConfig>(), c);
}

TEST(ConfigTest, OracleSatisfactionAlgorithmRoundTripsButFactoryRejects) {
    EXPECT_EQ(algorithmFromString("oracle_satisfaction"),
              RecommendationAlgorithm::OracleSatisfaction);
    EXPECT_STREQ(toString(RecommendationAlgorithm::OracleSatisfaction), "oracle_satisfaction");
}

// --- Realism V2 Phase 16: session-dynamics gate + block ---------------------------------------

TEST(ConfigTest, SessionDynamicsGateRequiresLatentReactions) {
    ExperimentConfig def;
    EXPECT_FALSE(def.realism.sessionDynamics);
    json ok = {{"realism",
                {{"content_v2", true}, {"latent_reactions", true}, {"session_dynamics", true}}}};
    EXPECT_TRUE(ok.get<ExperimentConfig>().realism.sessionDynamics);
    json bad = {{"realism", {{"content_v2", true}, {"session_dynamics", true}}}};
    EXPECT_THROW(bad.get<ExperimentConfig>(), std::invalid_argument);
}

TEST(ConfigTest, SessionDynamicsBlockParsesRoundTripsAndRejectsUnknownKeys) {
    ExperimentConfig def;
    EXPECT_DOUBLE_EQ(def.sessionDynamics.exitBias, -3.6);
    EXPECT_DOUBLE_EQ(def.sessionDynamics.regretLambda, 1.0);
    json j = {{"session_dynamics", {{"exit_bias", -2.0}, {"topic_fatigue_weight", 0.9}}}};
    auto c = j.get<ExperimentConfig>();
    EXPECT_DOUBLE_EQ(c.sessionDynamics.exitBias, -2.0);
    EXPECT_DOUBLE_EQ(c.sessionDynamics.topicFatigueWeight, 0.9);
    EXPECT_DOUBLE_EQ(c.sessionDynamics.awayDecayHalfLifeSeconds,
                     def.sessionDynamics.awayDecayHalfLifeSeconds);
    json out = c;
    EXPECT_EQ(out.get<ExperimentConfig>(), c);
    json bad = {{"session_dynamics", {{"exit_bais", -2.0}}}};
    EXPECT_THROW(bad.get<ExperimentConfig>(), std::invalid_argument);
}

// --- Realism V2 Phase 17: personalized-diversity gate + cohort mix ---------------------------

TEST(ConfigTest, PersonalizedDiversityGateRequiresSessionDynamics) {
    ExperimentConfig def;
    EXPECT_FALSE(def.realism.personalizedDiversity);
    EXPECT_TRUE(def.realism.cohortMix.empty());
    json ok = {{"realism",
                {{"content_v2", true},
                 {"latent_reactions", true},
                 {"session_dynamics", true},
                 {"personalized_diversity", true}}}};
    EXPECT_TRUE(ok.get<ExperimentConfig>().realism.personalizedDiversity);
    json bad = {
        {"realism",
         {{"content_v2", true}, {"latent_reactions", true}, {"personalized_diversity", true}}}};
    EXPECT_THROW(bad.get<ExperimentConfig>(), std::invalid_argument);
}

TEST(ConfigTest, CohortMixAndPersonalizedDiversityParamsParse) {
    json j = {{"realism", {{"cohort_mix", json::array({{{"name", "focused"}, {"weight", 2.0}}})}}},
              {"diversity", {{"personalized_cap_scale_max", 3.0}}}};
    auto c = j.get<ExperimentConfig>();
    ASSERT_EQ(c.realism.cohortMix.size(), 1u);
    EXPECT_EQ(c.realism.cohortMix[0].name, "focused");
    EXPECT_DOUBLE_EQ(c.diversity.personalizedCapScaleMax, 3.0);
    EXPECT_DOUBLE_EQ(c.diversity.personalizedLambdaMin, 0.60);
    json out = c;
    EXPECT_EQ(out.get<ExperimentConfig>(), c);
    json bad = {{"realism", {{"cohort_mix", json::array({{{"name", ""}, {"weight", 1.0}}})}}}};
    EXPECT_THROW(bad.get<ExperimentConfig>(), std::invalid_argument);
}

// --- Realism V2 Phase 18: scheduler + horizon + scheduling block ------------------------------

TEST(ConfigTest, SchedulerValidationAndSchedulingBlock) {
    ExperimentConfig def;
    EXPECT_EQ(def.simulation.scheduler, "round_robin");
    EXPECT_DOUBLE_EQ(def.simulation.horizonSeconds, 0.0);
    EXPECT_DOUBLE_EQ(def.scheduling.openStaggerSeconds, 43200.0);

    // event_queue requires horizon > 0 AND session_dynamics.
    json noHorizon = {
        {"simulation", {{"scheduler", "event_queue"}}},
        {"realism",
         {{"content_v2", true}, {"latent_reactions", true}, {"session_dynamics", true}}}};
    EXPECT_THROW(noHorizon.get<ExperimentConfig>(), std::invalid_argument);
    json noSessions = {{"simulation", {{"scheduler", "event_queue"}, {"horizon_seconds", 3600.0}}}};
    EXPECT_THROW(noSessions.get<ExperimentConfig>(), std::invalid_argument);
    json bogus = {{"simulation", {{"scheduler", "fifo"}}}};
    EXPECT_THROW(bogus.get<ExperimentConfig>(), std::invalid_argument);

    json ok = {
        {"simulation", {{"scheduler", "event_queue"}, {"horizon_seconds", 3600.0}}},
        {"realism", {{"content_v2", true}, {"latent_reactions", true}, {"session_dynamics", true}}},
        {"scheduling", {{"return_delay_mean_seconds", 7200.0}}}};
    auto c = ok.get<ExperimentConfig>();
    EXPECT_EQ(c.simulation.scheduler, "event_queue");
    EXPECT_DOUBLE_EQ(c.scheduling.returnDelayMeanSeconds, 7200.0);
    json out = c;
    EXPECT_EQ(out.get<ExperimentConfig>(), c);
}

TEST(ConfigTest, BehaviourV2BlockParsesRoundTripsAndRejectsUnknownKeys) {
    ExperimentConfig def;
    EXPECT_DOUBLE_EQ(def.behaviourV2.topicWeight, 1.5);
    EXPECT_DOUBLE_EQ(def.behaviourV2.latentNoiseStd, 0.30);

    json j = {{"behaviour_v2", {{"topic_weight", 2.0}, {"comment_propensity", 0.1}}}};
    auto c = j.get<ExperimentConfig>();
    EXPECT_DOUBLE_EQ(c.behaviourV2.topicWeight, 2.0);
    EXPECT_DOUBLE_EQ(c.behaviourV2.commentPropensity, 0.1);
    // Unspecified keys keep defaults.
    EXPECT_DOUBLE_EQ(c.behaviourV2.musicWeight, def.behaviourV2.musicWeight);

    json out = c;
    EXPECT_EQ(out.get<ExperimentConfig>(), c);

    json bad = {{"behaviour_v2", {{"topic_wight", 2.0}}}};
    EXPECT_THROW(bad.get<ExperimentConfig>(), std::invalid_argument);
}
