#include "rr/infrastructure/config.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace rr {

using nlohmann::json;

namespace {

void ensureKnownKeys(const json &j, const char *block,
                     std::initializer_list<const char *> allowed) {
    if (!j.is_object()) {
        throw std::invalid_argument(std::string("config block '") + block +
                                    "' must be a JSON object");
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        bool ok = false;
        for (const char *a : allowed) {
            if (it.key() == a) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            throw std::invalid_argument(std::string("unknown key '") + it.key() +
                                        "' in config block '" + block + "'");
        }
    }
}

template <class T> void readKey(const json &j, const char *key, T &out) {
    if (auto it = j.find(key); it != j.end()) {
        it->get_to(out);
    }
}

} // namespace

void to_json(json &j, const SimulationConfig &c) {
    j = json{{"seed", c.seed},
             {"users", c.users},
             {"reels", c.reels},
             {"creators", c.creators},
             {"topics", c.topics},
             {"dimensions", c.dimensions},
             {"interactions_per_user", c.interactionsPerUser},
             {"new_users", c.newUsers},
             {"new_users_at", c.newUsersAt},
             {"new_reels", c.newReels},
             {"new_reels_at", c.newReelsAt}};
}

void from_json(const json &j, SimulationConfig &c) {
    ensureKnownKeys(j, "simulation",
                    {"seed", "users", "reels", "creators", "topics", "dimensions",
                     "interactions_per_user", "new_users", "new_users_at", "new_reels",
                     "new_reels_at"});
    readKey(j, "seed", c.seed);
    readKey(j, "users", c.users);
    readKey(j, "reels", c.reels);
    readKey(j, "creators", c.creators);
    readKey(j, "topics", c.topics);
    readKey(j, "dimensions", c.dimensions);
    readKey(j, "interactions_per_user", c.interactionsPerUser);
    readKey(j, "new_users", c.newUsers);
    readKey(j, "new_users_at", c.newUsersAt);
    readKey(j, "new_reels", c.newReels);
    readKey(j, "new_reels_at", c.newReelsAt);
}

void to_json(json &j, const RecommendationConfig &c) {
    j = json{{"feed_size", c.feedSize},
             {"vector_candidates", c.vectorCandidates},
             {"popular_candidates", c.popularCandidates},
             {"fresh_candidates", c.freshCandidates},
             {"exploration_candidates", c.explorationCandidates},
             {"trending_candidates", c.trendingCandidates},
             {"creator_affinity_candidates", c.creatorAffinityCandidates}};
}

void from_json(const json &j, RecommendationConfig &c) {
    ensureKnownKeys(j, "recommendation",
                    {"feed_size", "vector_candidates", "popular_candidates", "fresh_candidates",
                     "exploration_candidates", "trending_candidates",
                     "creator_affinity_candidates"});
    readKey(j, "feed_size", c.feedSize);
    readKey(j, "vector_candidates", c.vectorCandidates);
    readKey(j, "popular_candidates", c.popularCandidates);
    readKey(j, "fresh_candidates", c.freshCandidates);
    readKey(j, "exploration_candidates", c.explorationCandidates);
    readKey(j, "trending_candidates", c.trendingCandidates);
    readKey(j, "creator_affinity_candidates", c.creatorAffinityCandidates);
}

void to_json(json &j, const HNSWConfig &c) {
    j = json{{"m", c.m}, {"ef_construction", c.efConstruction}, {"ef_search", c.efSearch}};
}

void from_json(const json &j, HNSWConfig &c) {
    ensureKnownKeys(j, "hnsw", {"m", "ef_construction", "ef_search"});
    readKey(j, "m", c.m);
    readKey(j, "ef_construction", c.efConstruction);
    readKey(j, "ef_search", c.efSearch);
}

void to_json(json &j, const RankingConfig &c) {
    j = json{{"similarity_weight", c.similarityWeight},
             {"quality_weight", c.qualityWeight},
             {"freshness_weight", c.freshnessWeight},
             {"popularity_weight", c.popularityWeight},
             {"trending_weight", c.trendingWeight},
             {"creator_affinity_weight", c.creatorAffinityWeight},
             {"exploration_weight", c.explorationWeight},
             {"repetition_penalty", c.repetitionPenalty},
             {"duration_match_weight", c.durationMatchWeight},
             {"impression_penalty_weight", c.impressionPenaltyWeight},
             {"session_topic_weight", c.sessionTopicWeight},
             {"freshness_half_life_seconds", c.freshnessHalfLifeSeconds},
             {"trending_half_life_seconds", c.trendingHalfLifeSeconds}};
}

void from_json(const json &j, RankingConfig &c) {
    ensureKnownKeys(j, "ranking",
                    {"similarity_weight", "quality_weight", "freshness_weight", "popularity_weight",
                     "trending_weight", "creator_affinity_weight", "exploration_weight",
                     "repetition_penalty", "duration_match_weight", "impression_penalty_weight",
                     "session_topic_weight", "freshness_half_life_seconds",
                     "trending_half_life_seconds"});
    readKey(j, "similarity_weight", c.similarityWeight);
    readKey(j, "quality_weight", c.qualityWeight);
    readKey(j, "freshness_weight", c.freshnessWeight);
    readKey(j, "popularity_weight", c.popularityWeight);
    readKey(j, "trending_weight", c.trendingWeight);
    readKey(j, "creator_affinity_weight", c.creatorAffinityWeight);
    readKey(j, "exploration_weight", c.explorationWeight);
    readKey(j, "repetition_penalty", c.repetitionPenalty);
    readKey(j, "duration_match_weight", c.durationMatchWeight);
    readKey(j, "impression_penalty_weight", c.impressionPenaltyWeight);
    readKey(j, "session_topic_weight", c.sessionTopicWeight);
    readKey(j, "freshness_half_life_seconds", c.freshnessHalfLifeSeconds);
    readKey(j, "trending_half_life_seconds", c.trendingHalfLifeSeconds);
}

void to_json(json &j, const LearningConfig &c) {
    j = json{{"enabled", c.enabled},
             {"long_term_rate", c.longTermRate},
             {"session_rate", c.sessionRate},
             {"recent_window", c.recentWindow},
             {"session_lambda", c.sessionLambda},
             {"long_term_weight", c.longTermWeight},
             {"session_weight", c.sessionWeight}};
}

void from_json(const json &j, LearningConfig &c) {
    ensureKnownKeys(j, "learning",
                    {"enabled", "long_term_rate", "session_rate", "recent_window", "session_lambda",
                     "long_term_weight", "session_weight"});
    readKey(j, "enabled", c.enabled);
    readKey(j, "long_term_rate", c.longTermRate);
    readKey(j, "session_rate", c.sessionRate);
    readKey(j, "recent_window", c.recentWindow);
    readKey(j, "session_lambda", c.sessionLambda);
    readKey(j, "long_term_weight", c.longTermWeight);
    readKey(j, "session_weight", c.sessionWeight);
}

void to_json(json &j, const ExplorationConfig &c) {
    j = json{{"enabled", c.enabled},
             {"epsilon", c.epsilon},
             {"fresh_window_seconds", c.freshWindowSeconds},
             {"guaranteed_slots", c.guaranteedSlots}};
}

void from_json(const json &j, ExplorationConfig &c) {
    ensureKnownKeys(j, "exploration",
                    {"enabled", "epsilon", "fresh_window_seconds", "guaranteed_slots"});
    readKey(j, "enabled", c.enabled);
    readKey(j, "epsilon", c.epsilon);
    readKey(j, "fresh_window_seconds", c.freshWindowSeconds);
    readKey(j, "guaranteed_slots", c.guaranteedSlots);
}

void to_json(json &j, const DiversityConfig &c) {
    j = json{{"enabled", c.enabled},
             {"max_per_creator", c.maxPerCreator},
             {"max_per_topic", c.maxPerTopic},
             {"mmr_lambda", c.mmrLambda},
             {"use_mmr", c.useMmr}};
}

void from_json(const json &j, DiversityConfig &c) {
    ensureKnownKeys(j, "diversity",
                    {"enabled", "max_per_creator", "max_per_topic", "mmr_lambda", "use_mmr"});
    readKey(j, "enabled", c.enabled);
    readKey(j, "max_per_creator", c.maxPerCreator);
    readKey(j, "max_per_topic", c.maxPerTopic);
    readKey(j, "mmr_lambda", c.mmrLambda);
    readKey(j, "use_mmr", c.useMmr);
}

void to_json(json &j, const DriftTopicWeight &c) {
    j = json{{"topic", c.topic}, {"weight", c.weight}};
}

void from_json(const json &j, DriftTopicWeight &c) {
    ensureKnownKeys(j, "drift.events[].topic_mix[]", {"topic", "weight"});
    readKey(j, "topic", c.topic);
    readKey(j, "weight", c.weight);
}

void to_json(json &j, const DriftEvent &c) {
    j = json{{"at_interaction", c.atInteraction},
             {"cohort_lo", c.cohortLo},
             {"cohort_hi", c.cohortHi},
             {"topic_mix", c.topicMix}};
}

void from_json(const json &j, DriftEvent &c) {
    ensureKnownKeys(j, "drift.events[]", {"at_interaction", "cohort_lo", "cohort_hi", "topic_mix"});
    readKey(j, "at_interaction", c.atInteraction);
    readKey(j, "cohort_lo", c.cohortLo);
    readKey(j, "cohort_hi", c.cohortHi);
    readKey(j, "topic_mix", c.topicMix);
}

void to_json(json &j, const DriftConfig &c) { j = json{{"events", c.events}}; }

void from_json(const json &j, DriftConfig &c) {
    ensureKnownKeys(j, "drift", {"events"});
    readKey(j, "events", c.events);
}

void to_json(json &j, const BehaviourConfig &c) {
    j = json{{"alpha", c.alpha},
             {"beta", c.beta},
             {"gamma", c.gamma},
             {"delta", c.delta},
             {"noise_std", c.noiseStd},
             {"skip_bias", c.skipBias},
             {"not_interested_z", c.notInterestedZ},
             {"not_interested_prob", c.notInterestedProb}};
}

void from_json(const json &j, BehaviourConfig &c) {
    ensureKnownKeys(j, "behaviour",
                    {"alpha", "beta", "gamma", "delta", "noise_std", "skip_bias",
                     "not_interested_z", "not_interested_prob"});
    readKey(j, "alpha", c.alpha);
    readKey(j, "beta", c.beta);
    readKey(j, "gamma", c.gamma);
    readKey(j, "delta", c.delta);
    readKey(j, "noise_std", c.noiseStd);
    readKey(j, "skip_bias", c.skipBias);
    readKey(j, "not_interested_z", c.notInterestedZ);
    readKey(j, "not_interested_prob", c.notInterestedProb);
}

void to_json(json &j, const RewardConfig &c) {
    j = json{{"watch_ratio_weight", c.watchRatioWeight},
             {"watch_seconds_weight", c.watchSecondsWeight},
             {"like_weight", c.likeWeight},
             {"share_weight", c.shareWeight},
             {"follow_weight", c.followWeight},
             {"instant_skip_penalty", c.instantSkipPenalty},
             {"not_interested_penalty", c.notInterestedPenalty}};
}

void from_json(const json &j, RewardConfig &c) {
    ensureKnownKeys(j, "reward",
                    {"watch_ratio_weight", "watch_seconds_weight", "like_weight", "share_weight",
                     "follow_weight", "instant_skip_penalty", "not_interested_penalty"});
    readKey(j, "watch_ratio_weight", c.watchRatioWeight);
    readKey(j, "watch_seconds_weight", c.watchSecondsWeight);
    readKey(j, "like_weight", c.likeWeight);
    readKey(j, "share_weight", c.shareWeight);
    readKey(j, "follow_weight", c.followWeight);
    readKey(j, "instant_skip_penalty", c.instantSkipPenalty);
    readKey(j, "not_interested_penalty", c.notInterestedPenalty);
}

void to_json(json &j, const EvaluationConfig &c) {
    j = json{{"oracle_sample_rate", c.oracleSampleRate},
             {"retrieval_sample_rate", c.retrievalSampleRate}};
}

void from_json(const json &j, EvaluationConfig &c) {
    ensureKnownKeys(j, "evaluation", {"oracle_sample_rate", "retrieval_sample_rate"});
    readKey(j, "oracle_sample_rate", c.oracleSampleRate);
    readKey(j, "retrieval_sample_rate", c.retrievalSampleRate);
}

const char *toString(RecommendationAlgorithm a) {
    switch (a) {
    case RecommendationAlgorithm::Random:
        return "random";
    case RecommendationAlgorithm::Popularity:
        return "popularity";
    case RecommendationAlgorithm::ExactVector:
        return "exact_vector";
    case RecommendationAlgorithm::Hnsw:
        return "hnsw";
    case RecommendationAlgorithm::HnswRanker:
        return "hnsw_ranker";
    case RecommendationAlgorithm::HnswRankerDiversity:
        return "hnsw_ranker_diversity";
    case RecommendationAlgorithm::HnswRankerExploration:
        return "hnsw_ranker_exploration";
    }
    return "random";
}

RecommendationAlgorithm algorithmFromString(const std::string &s) {
    if (s == "random") {
        return RecommendationAlgorithm::Random;
    }
    if (s == "popularity") {
        return RecommendationAlgorithm::Popularity;
    }
    if (s == "exact_vector") {
        return RecommendationAlgorithm::ExactVector;
    }
    if (s == "hnsw") {
        return RecommendationAlgorithm::Hnsw;
    }
    if (s == "hnsw_ranker") {
        return RecommendationAlgorithm::HnswRanker;
    }
    if (s == "hnsw_ranker_diversity") {
        return RecommendationAlgorithm::HnswRankerDiversity;
    }
    if (s == "hnsw_ranker_exploration") {
        return RecommendationAlgorithm::HnswRankerExploration;
    }
    throw std::invalid_argument(
        "unknown algorithm '" + s +
        "'; valid values: random, popularity, exact_vector, hnsw, hnsw_ranker, "
        "hnsw_ranker_diversity, hnsw_ranker_exploration");
}

void to_json(json &j, const RecommendationAlgorithm &a) { j = toString(a); }

void from_json(const json &j, RecommendationAlgorithm &a) {
    a = algorithmFromString(j.get<std::string>());
}

void to_json(json &j, const RealismConfig &c) {
    j = json{{"content_v2", c.contentV2}, {"languages", c.languages}, {"archetypes", c.archetypes}};
}

void from_json(const json &j, RealismConfig &c) {
    ensureKnownKeys(j, "realism", {"content_v2", "languages", "archetypes"});
    readKey(j, "content_v2", c.contentV2);
    readKey(j, "languages", c.languages);
    readKey(j, "archetypes", c.archetypes);
    if (c.languages == 0) {
        throw std::invalid_argument("realism.languages must be >= 1");
    }
    if (c.archetypes.empty()) {
        throw std::invalid_argument("realism.archetypes must not be empty");
    }
}

void to_json(json &j, const ExperimentConfig &c) {
    j = json{{"simulation", c.simulation},
             {"recommendation", c.recommendation},
             {"algorithm", toString(c.algorithm)},
             {"hnsw", c.hnsw},
             {"ranking", c.ranking},
             {"learning", c.learning},
             {"exploration", c.exploration},
             {"diversity", c.diversity},
             {"drift", c.drift},
             {"behaviour", c.behaviour},
             {"reward", c.reward},
             {"evaluation", c.evaluation},
             {"realism", c.realism}};
}

void from_json(const json &j, ExperimentConfig &c) {
    ensureKnownKeys(j, "<top-level>",
                    {"simulation", "recommendation", "algorithm", "hnsw", "ranking", "learning",
                     "exploration", "diversity", "drift", "behaviour", "reward", "evaluation",
                     "realism"});
    readKey(j, "simulation", c.simulation);
    readKey(j, "recommendation", c.recommendation);
    readKey(j, "algorithm", c.algorithm);
    readKey(j, "hnsw", c.hnsw);
    readKey(j, "ranking", c.ranking);
    readKey(j, "learning", c.learning);
    readKey(j, "exploration", c.exploration);
    readKey(j, "diversity", c.diversity);
    readKey(j, "drift", c.drift);
    readKey(j, "behaviour", c.behaviour);
    readKey(j, "reward", c.reward);
    readKey(j, "evaluation", c.evaluation);
    readKey(j, "realism", c.realism);
    // Phase 13 scope guard: mid-simulation injection (Phase 8) predates the V2 content model and
    // would need V2 augmentation of injected entities on the injection streams — unsupported
    // until a phase needs it. Fail fast at load rather than silently generating injected
    // entities with default V2 fields.
    if (c.realism.contentV2 && (c.simulation.newUsers > 0 || c.simulation.newReels > 0)) {
        throw std::invalid_argument("realism.content_v2 does not support mid-simulation "
                                    "injection (simulation.new_users/new_reels) yet");
    }
}

ExperimentConfig loadExperimentConfig(const std::filesystem::path &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument("cannot open config file: " + path.string());
    }
    json j;
    try {
        in >> j;
    } catch (const json::parse_error &e) {
        throw std::invalid_argument("failed to parse config file '" + path.string() +
                                    "': " + e.what());
    }
    return j.get<ExperimentConfig>();
}

} // namespace rr
