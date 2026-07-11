#pragma once

#include <cstdint>
#include <filesystem>

#include <nlohmann/json_fwd.hpp>

namespace rr {

// Every struct default-constructs to the TDD 21 suggested values. JSON keys are snake_case
// exactly as the TDD; struct members are camelCase (design decision D10). Unknown JSON keys are
// an error; missing keys keep the default (D6).

struct SimulationConfig {
    uint64_t seed = 42;
    uint32_t users = 10000;
    uint32_t reels = 100000;
    uint32_t creators = 5000;
    uint32_t topics = 32;
    uint32_t dimensions = 64;
    uint32_t interactionsPerUser = 200;
    bool operator==(const SimulationConfig &) const = default;
};

struct RecommendationConfig {
    uint32_t feedSize = 10;
    uint32_t vectorCandidates = 500;
    uint32_t popularCandidates = 100;
    uint32_t freshCandidates = 100;
    uint32_t explorationCandidates = 50;
    bool operator==(const RecommendationConfig &) const = default;
};

struct HNSWConfig {
    uint32_t m = 16;
    uint32_t efConstruction = 200;
    uint32_t efSearch = 64;
    bool operator==(const HNSWConfig &) const = default;
};

struct RankingConfig {
    double similarityWeight = 0.50;
    double qualityWeight = 0.10;
    double freshnessWeight = 0.08;
    double popularityWeight = 0.07;
    double trendingWeight = 0.08;
    double creatorAffinityWeight = 0.07;
    double explorationWeight = 0.05;
    double repetitionPenalty = 0.15;
    bool operator==(const RankingConfig &) const = default;
};

struct LearningConfig {
    double longTermRate = 0.02;
    double sessionRate = 0.15;
    uint32_t recentWindow = 20;
    // long_term_weight / session_weight are an addition to the TDD 21 example, mandated
    // configurable by TDD 8.3.
    double longTermWeight = 0.65;
    double sessionWeight = 0.35;
    bool operator==(const LearningConfig &) const = default;
};

struct ExplorationConfig {
    bool enabled = true;
    double epsilon = 0.05;
    bool operator==(const ExplorationConfig &) const = default;
};

// TDD 10.2/10.3 behaviour-score parameters: z = alpha*a + beta*Q + gamma*C - delta*D + eps,
// P(complete) = sigmoid(z), P(instantSkip) = sigmoid(-z + skipBias). The TDD mandates these
// configurable but suggests no values; defaults here are calibrated so the Phase 3 statistical
// monotonicity tests pass at TDD-default data-generation settings (tests must not assert these
// exact values — they are tuning surface, not contract).
struct BehaviourConfig {
    double alpha = 4.0;              // weight on base affinity a = p_u . q_v            (TDD 10.1)
    double beta = 1.0;               // weight on intrinsic quality Q_v
    double gamma = 0.5;              // weight on true creator affinity C_{u,c}
    double delta = 1.0;              // weight on duration penalty D_v
    double noiseStd = 0.5;           // stddev of the gaussian noise term eps
    double skipBias = 1.0;           // b in P(instantSkip) = sigmoid(-z + b)
    double notInterestedZ = -2.0;    // NotInterested is possible only when z < this
    double notInterestedProb = 0.05; // ... and then fires with this probability
    bool operator==(const BehaviourConfig &) const = default;
};

// TDD 10.5 reward weights (defaults are the TDD's exact suggested values; the reward itself is
// clamped to [-1, 1]).
struct RewardConfig {
    double watchRatioWeight = 0.45;
    double watchSecondsWeight = 0.15; // applied to log(1 + watchSeconds), normalized
    double likeWeight = 0.15;
    double shareWeight = 0.20;
    double followWeight = 0.15;
    double instantSkipPenalty = 0.35;
    double notInterestedPenalty = 0.75;
    bool operator==(const RewardConfig &) const = default;
};

struct DiversityConfig {
    bool enabled = true;
    uint32_t maxPerCreator = 2;
    uint32_t maxPerTopic = 3;
    double mmrLambda = 0.75;
    bool operator==(const DiversityConfig &) const = default;
};

// Evaluation-harness parameters (TDD 19 / phase-4 task 5). The oracle exhaustively scores all
// reels by true hidden affinity, so it runs only on a Bernoulli-sampled subset of requests; the
// rate is config-driven and recorded in every experiment's output.
struct EvaluationConfig {
    double oracleSampleRate = 0.05;
    // TDD 18.1 live retrieval metrics (Phase 5): fraction of requests on which the harness
    // compares the recommender's vector index against exact ground truth (Recall@K, distance
    // error). Exact search over the full corpus per sample keeps this a sampled measurement.
    double retrievalSampleRate = 0.02;
    bool operator==(const EvaluationConfig &) const = default;
};

// TDD 16.1-16.7.
enum class RecommendationAlgorithm {
    Random,
    Popularity,
    ExactVector,
    Hnsw,
    HnswRanker,
    HnswRankerDiversity,
    HnswRankerExploration,
};

struct ExperimentConfig {
    SimulationConfig simulation;
    RecommendationConfig recommendation;
    RecommendationAlgorithm algorithm = RecommendationAlgorithm::Random;
    HNSWConfig hnsw;
    RankingConfig ranking;
    LearningConfig learning;
    ExplorationConfig exploration;
    DiversityConfig diversity;
    BehaviourConfig behaviour;
    RewardConfig reward;
    EvaluationConfig evaluation;
    bool operator==(const ExperimentConfig &) const = default;
};

// Serialization (defined in config.cpp). from_json throws std::invalid_argument on unknown keys.
void to_json(nlohmann::json &j, const SimulationConfig &c);
void from_json(const nlohmann::json &j, SimulationConfig &c);
void to_json(nlohmann::json &j, const RecommendationConfig &c);
void from_json(const nlohmann::json &j, RecommendationConfig &c);
void to_json(nlohmann::json &j, const HNSWConfig &c);
void from_json(const nlohmann::json &j, HNSWConfig &c);
void to_json(nlohmann::json &j, const RankingConfig &c);
void from_json(const nlohmann::json &j, RankingConfig &c);
void to_json(nlohmann::json &j, const LearningConfig &c);
void from_json(const nlohmann::json &j, LearningConfig &c);
void to_json(nlohmann::json &j, const ExplorationConfig &c);
void from_json(const nlohmann::json &j, ExplorationConfig &c);
void to_json(nlohmann::json &j, const DiversityConfig &c);
void from_json(const nlohmann::json &j, DiversityConfig &c);
void to_json(nlohmann::json &j, const BehaviourConfig &c);
void from_json(const nlohmann::json &j, BehaviourConfig &c);
void to_json(nlohmann::json &j, const RewardConfig &c);
void from_json(const nlohmann::json &j, RewardConfig &c);
void to_json(nlohmann::json &j, const EvaluationConfig &c);
void from_json(const nlohmann::json &j, EvaluationConfig &c);
void to_json(nlohmann::json &j, const RecommendationAlgorithm &a);
void from_json(const nlohmann::json &j, RecommendationAlgorithm &a);
void to_json(nlohmann::json &j, const ExperimentConfig &c);
void from_json(const nlohmann::json &j, ExperimentConfig &c);

// Map an algorithm to/from its canonical string. algorithmFromString throws
// std::invalid_argument (listing valid values) on an unknown name.
const char *toString(RecommendationAlgorithm a);
RecommendationAlgorithm algorithmFromString(const std::string &s);

// Load and parse a config file. Throws std::invalid_argument (with the path in the message) on a
// missing/unreadable file or a parse error.
ExperimentConfig loadExperimentConfig(const std::filesystem::path &path);

} // namespace rr
