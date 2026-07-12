#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

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
    // Mid-simulation entity injection (Phase 8, TDD 18.5 cold-start metrics): at the START of
    // round new_users_at / new_reels_at (0-based round index in the harness's interleaved loop),
    // inject that many freshly generated users / reels so genuinely-cold entities are measurable.
    // A count of 0 disables the injection (the "_at" value is then ignored). Injected entities are
    // generated on their own named rng streams, so enabling injection never perturbs the original
    // dataset (D8). An addition to the TDD 21 example, mandated by phase-8 task 5.
    uint32_t newUsers = 0;
    uint32_t newUsersAt = 0;
    uint32_t newReels = 0;
    uint32_t newReelsAt = 0;
    bool operator==(const SimulationConfig &) const = default;
};

struct RecommendationConfig {
    uint32_t feedSize = 10;
    uint32_t vectorCandidates = 500;
    uint32_t popularCandidates = 100;
    uint32_t freshCandidates = 100;
    uint32_t explorationCandidates = 50;
    // trending_candidates / creator_affinity_candidates are an addition to the TDD 21 example;
    // the counts themselves are mandated by TDD 13 (Trending: 100, Creator affinity: 100).
    uint32_t trendingCandidates = 100;
    uint32_t creatorAffinityCandidates = 100;
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
    // duration_match_weight / impression_penalty_weight are an addition to the TDD 14.2 example
    // formula: 14.1 mandates duration-preference match and previous-impression count as ranking
    // features, and 14.2 mandates every weight be configuration-driven. Duration match is a
    // bonus; the impression term is a fatigue penalty (subtracted, like repetition_penalty).
    double durationMatchWeight = 0.05;
    double impressionPenaltyWeight = 0.05;
    // Weight on the TDD 14.1 session-topic-similarity feature (cosine between the candidate and
    // the user's session preference vector), active from Phase 7 when session vectors start
    // updating online. An addition to the TDD 14.2 example weights (which predate the feature
    // list's session term); config-driven like every other weight per 14.2.
    double sessionTopicWeight = 0.05;
    // Decay half-lives (simulated seconds) for the two time-sensitive scoring inputs, an addition
    // to the TDD 21 example: freshness decay from createdAt is mandated configurable by TDD 8.2,
    // trending time-decay by TDD 12.4. Consumed by rr::freshnessScore / rr::trendingScore and by
    // Simulator::step's accumulator maintenance.
    double freshnessHalfLifeSeconds = 604800.0; // 7 simulated days
    double trendingHalfLifeSeconds = 21600.0;   // 6 simulated hours
    bool operator==(const RankingConfig &) const = default;
};

struct LearningConfig {
    // Master switch for online preference learning (Phase 7): false freezes every user at the
    // cold-start estimate (the pre-Phase-7 behaviour), giving the learning-vs-frozen experiment
    // arm mandated by the phase plan. An addition to the TDD 21 example.
    bool enabled = true;
    double longTermRate = 0.02;
    double sessionRate = 0.15;
    uint32_t recentWindow = 20;
    // TDD 11.3 session-decay lambda (suggested range 0.85-0.95): weight lambda^(n-i) applied to
    // the i-th most recent interaction when recomputing the session vector. An addition to the
    // TDD 21 example, mandated configurable by TDD 11.3.
    double sessionLambda = 0.90;
    // long_term_weight / session_weight are an addition to the TDD 21 example, mandated
    // configurable by TDD 8.3.
    double longTermWeight = 0.65;
    double sessionWeight = 0.35;
    bool operator==(const LearningConfig &) const = default;
};

struct ExplorationConfig {
    bool enabled = true;
    double epsilon = 0.05;
    // Recency window (simulated seconds) defining a "recently created" reel for the Fresh
    // candidate source (TDD 12.5) and the exploration source's random-fresh mode (TDD 12.7):
    // a reel qualifies at request time t iff createdAt >= t - fresh_window_seconds. An addition
    // to the TDD 21 example; TDD 12.5 defines no window, and experiments need to vary it.
    double freshWindowSeconds = 259200.0; // 3 simulated days
    // Phase-8 task 3: exploration candidates are protected from being ranked out entirely. The
    // final feed carries min(slots that fired the per-slot epsilon gate this request,
    // guaranteed_slots, exploration candidates available in the ranked pool) exploration items;
    // the promotion rule is documented at its implementation in the Orchestrator. epsilon = 0
    // fires no slots, so the guarantee is structurally inert there (the epsilon=0 no-op exit
    // criterion).
    uint32_t guaranteedSlots = 2;
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
    // Phase 9: selects between the two diversity re-ranking modes the experiments compare —
    // constraints only (false) vs constraints + MMR ordering within them (true, TDD 15.2). An
    // addition to the TDD 21 example, mandated by the phase-9 experiment arms (hnsw_ranker vs
    // +constraints vs +constraints+MMR).
    bool useMmr = true;
    bool operator==(const DiversityConfig &) const = default;
};

// One weighted topic in a drift event's target mix (TDD 11.4). `topic` is a TopicId value that
// must exist in the generated topic set (validated by DriftScheduler at construction); `weight` is
// a relative weight > 0 — the mix is normalized at application, so weights need not sum to 1.
struct DriftTopicWeight {
    uint32_t topic = 0;
    double weight = 0.0;
    bool operator==(const DriftTopicWeight &) const = default;
};

// One scheduled hidden-preference change (TDD 11.4, Phase 10): when a user in the cohort reaches
// `atInteraction` completed interactions, their hidden preference is rebuilt from `topicMix`
// (interactions with 0-based index >= atInteraction see the new preference). The cohort is the
// deterministic, rng-free user slice with hash01(userId) in [cohortLo, cohortHi) — disjoint
// [lo, hi) ranges express disjoint cohorts drifting to different mixes. Defaults cover the whole
// population.
struct DriftEvent {
    uint32_t atInteraction = 0;
    double cohortLo = 0.0;
    double cohortHi = 1.0;
    std::vector<DriftTopicWeight> topicMix;
    bool operator==(const DriftEvent &) const = default;
};

// Scheduled interest drift (TDD 11.4, Phase 10). Empty `events` = drift disabled: every drift
// branch in the simulator/harness is inert and the run's output is byte-identical to a
// pre-Phase-10 run (the regression contract, same pattern as Phase 8 injection). An addition to
// the TDD 21 example, mandated by phase-10 task 1.
struct DriftConfig {
    std::vector<DriftEvent> events;
    bool operator==(const DriftConfig &) const = default;
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
    DriftConfig drift;
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
void to_json(nlohmann::json &j, const DriftTopicWeight &c);
void from_json(const nlohmann::json &j, DriftTopicWeight &c);
void to_json(nlohmann::json &j, const DriftEvent &c);
void from_json(const nlohmann::json &j, DriftEvent &c);
void to_json(nlohmann::json &j, const DriftConfig &c);
void from_json(const nlohmann::json &j, DriftConfig &c);
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
