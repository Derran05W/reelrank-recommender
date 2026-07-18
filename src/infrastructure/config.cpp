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
             {"trending_half_life_seconds", c.trendingHalfLifeSeconds},
             {"visual_match_weight", c.visualMatchWeight},
             {"music_match_weight", c.musicMatchWeight},
             {"emotional_match_weight", c.emotionalMatchWeight},
             {"clickbait_weight", c.clickbaitWeight},
             {"emotional_intensity_weight", c.emotionalIntensityWeight},
             {"usefulness_weight", c.usefulnessWeight},
             {"production_quality_weight", c.productionQualityWeight},
             {"information_density_weight", c.informationDensityWeight},
             {"language_match_weight", c.languageMatchWeight},
             {"save_popularity_weight", c.savePopularityWeight}};
}

void from_json(const json &j, RankingConfig &c) {
    ensureKnownKeys(j, "ranking",
                    {"similarity_weight",
                     "quality_weight",
                     "freshness_weight",
                     "popularity_weight",
                     "trending_weight",
                     "creator_affinity_weight",
                     "exploration_weight",
                     "repetition_penalty",
                     "duration_match_weight",
                     "impression_penalty_weight",
                     "session_topic_weight",
                     "freshness_half_life_seconds",
                     "trending_half_life_seconds",
                     "visual_match_weight",
                     "music_match_weight",
                     "emotional_match_weight",
                     "clickbait_weight",
                     "emotional_intensity_weight",
                     "usefulness_weight",
                     "production_quality_weight",
                     "information_density_weight",
                     "language_match_weight",
                     "save_popularity_weight"});
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
    readKey(j, "visual_match_weight", c.visualMatchWeight);
    readKey(j, "music_match_weight", c.musicMatchWeight);
    readKey(j, "emotional_match_weight", c.emotionalMatchWeight);
    readKey(j, "clickbait_weight", c.clickbaitWeight);
    readKey(j, "emotional_intensity_weight", c.emotionalIntensityWeight);
    readKey(j, "usefulness_weight", c.usefulnessWeight);
    readKey(j, "production_quality_weight", c.productionQualityWeight);
    readKey(j, "information_density_weight", c.informationDensityWeight);
    readKey(j, "language_match_weight", c.languageMatchWeight);
    readKey(j, "save_popularity_weight", c.savePopularityWeight);
}

void to_json(json &j, const LearningConfig &c) {
    j = json{{"enabled", c.enabled},
             {"long_term_rate", c.longTermRate},
             {"session_rate", c.sessionRate},
             {"recent_window", c.recentWindow},
             {"session_lambda", c.sessionLambda},
             {"long_term_weight", c.longTermWeight},
             {"session_weight", c.sessionWeight},
             {"modality_rate", c.modalityRate}};
}

void from_json(const json &j, LearningConfig &c) {
    ensureKnownKeys(j, "learning",
                    {"enabled", "long_term_rate", "session_rate", "recent_window", "session_lambda",
                     "long_term_weight", "session_weight", "modality_rate"});
    readKey(j, "enabled", c.enabled);
    readKey(j, "long_term_rate", c.longTermRate);
    readKey(j, "session_rate", c.sessionRate);
    readKey(j, "recent_window", c.recentWindow);
    readKey(j, "session_lambda", c.sessionLambda);
    readKey(j, "long_term_weight", c.longTermWeight);
    readKey(j, "session_weight", c.sessionWeight);
    readKey(j, "modality_rate", c.modalityRate);
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
    case RecommendationAlgorithm::OracleSatisfaction:
        return "oracle_satisfaction";
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
    if (s == "oracle_satisfaction") {
        return RecommendationAlgorithm::OracleSatisfaction;
    }
    throw std::invalid_argument(
        "unknown algorithm '" + s +
        "'; valid values: random, popularity, exact_vector, hnsw, hnsw_ranker, "
        "hnsw_ranker_diversity, hnsw_ranker_exploration, oracle_satisfaction");
}

void to_json(json &j, const RecommendationAlgorithm &a) { j = toString(a); }

void from_json(const json &j, RecommendationAlgorithm &a) {
    a = algorithmFromString(j.get<std::string>());
}

void to_json(json &j, const RealismConfig &c) {
    j = json{{"content_v2", c.contentV2},
             {"latent_reactions", c.latentReactions},
             {"session_dynamics", c.sessionDynamics},
             {"languages", c.languages},
             {"archetypes", c.archetypes}};
}

void from_json(const json &j, RealismConfig &c) {
    ensureKnownKeys(
        j, "realism",
        {"content_v2", "latent_reactions", "session_dynamics", "languages", "archetypes"});
    readKey(j, "content_v2", c.contentV2);
    readKey(j, "latent_reactions", c.latentReactions);
    readKey(j, "session_dynamics", c.sessionDynamics);
    readKey(j, "languages", c.languages);
    readKey(j, "archetypes", c.archetypes);
    if (c.languages == 0) {
        throw std::invalid_argument("realism.languages must be >= 1");
    }
    if (c.archetypes.empty()) {
        throw std::invalid_argument("realism.archetypes must not be empty");
    }
    // D17 gate dependencies: latent reactions consume the V2 content/user factor model;
    // session dynamics consume the latent stream (fatigueDelta, satisfaction, regret).
    if (c.latentReactions && !c.contentV2) {
        throw std::invalid_argument("realism.latent_reactions requires realism.content_v2");
    }
    if (c.sessionDynamics && !c.latentReactions) {
        throw std::invalid_argument("realism.session_dynamics requires realism.latent_reactions");
    }
}

void to_json(json &j, const BehaviourV2Config &c) {
    j = json{{"topic_weight", c.topicWeight},
             {"visual_weight", c.visualWeight},
             {"music_weight", c.musicWeight},
             {"emotional_weight", c.emotionalWeight},
             {"usefulness_weight", c.usefulnessWeight},
             {"humour_weight", c.humourWeight},
             {"novelty_weight", c.noveltyWeight},
             {"information_density_weight", c.informationDensityWeight},
             {"controversy_penalty_weight", c.controversyPenaltyWeight},
             {"controversy_boost_weight", c.controversyBoostWeight},
             {"language_mismatch_penalty", c.languageMismatchPenalty},
             {"creator_attachment_weight", c.creatorAttachmentWeight},
             {"latent_noise_std", c.latentNoiseStd},
             {"comment_propensity", c.commentPropensity},
             {"save_propensity", c.savePropensity},
             {"profile_visit_propensity", c.profileVisitPropensity},
             {"social_conformity_weight", c.socialConformityWeight},
             {"short_completion_boost", c.shortCompletionBoost},
             {"short_duration_seconds", c.shortDurationSeconds}};
}

void from_json(const json &j, BehaviourV2Config &c) {
    ensureKnownKeys(j, "behaviour_v2",
                    {"topic_weight", "visual_weight", "music_weight", "emotional_weight",
                     "usefulness_weight", "humour_weight", "novelty_weight",
                     "information_density_weight", "controversy_penalty_weight",
                     "controversy_boost_weight", "language_mismatch_penalty",
                     "creator_attachment_weight", "latent_noise_std", "comment_propensity",
                     "save_propensity", "profile_visit_propensity", "social_conformity_weight",
                     "short_completion_boost", "short_duration_seconds"});
    readKey(j, "topic_weight", c.topicWeight);
    readKey(j, "visual_weight", c.visualWeight);
    readKey(j, "music_weight", c.musicWeight);
    readKey(j, "emotional_weight", c.emotionalWeight);
    readKey(j, "usefulness_weight", c.usefulnessWeight);
    readKey(j, "humour_weight", c.humourWeight);
    readKey(j, "novelty_weight", c.noveltyWeight);
    readKey(j, "information_density_weight", c.informationDensityWeight);
    readKey(j, "controversy_penalty_weight", c.controversyPenaltyWeight);
    readKey(j, "controversy_boost_weight", c.controversyBoostWeight);
    readKey(j, "language_mismatch_penalty", c.languageMismatchPenalty);
    readKey(j, "creator_attachment_weight", c.creatorAttachmentWeight);
    readKey(j, "latent_noise_std", c.latentNoiseStd);
    readKey(j, "comment_propensity", c.commentPropensity);
    readKey(j, "save_propensity", c.savePropensity);
    readKey(j, "profile_visit_propensity", c.profileVisitPropensity);
    readKey(j, "social_conformity_weight", c.socialConformityWeight);
    readKey(j, "short_completion_boost", c.shortCompletionBoost);
    readKey(j, "short_duration_seconds", c.shortDurationSeconds);
}

void to_json(json &j, const SessionDynamicsConfig &c) {
    j = json{{"topic_fatigue_weight", c.topicFatigueWeight},
             {"creator_fatigue_weight", c.creatorFatigueWeight},
             {"novelty_match_weight", c.noveltyMatchWeight},
             {"topic_fatigue_increment", c.topicFatigueIncrement},
             {"creator_fatigue_increment", c.creatorFatigueIncrement},
             {"general_fatigue_scale", c.generalFatigueScale},
             {"away_decay_half_life_seconds", c.awayDecayHalfLifeSeconds},
             {"exit_bias", c.exitBias},
             {"exit_fatigue_weight", c.exitFatigueWeight},
             {"exit_regret_weight", c.exitRegretWeight},
             {"exit_poor_streak_weight", c.exitPoorStreakWeight},
             {"exit_satisfaction_weight", c.exitSatisfactionWeight},
             {"exit_interruption_weight", c.exitInterruptionWeight},
             {"external_interruption_hazard", c.externalInterruptionHazard},
             {"regret_lambda", c.regretLambda},
             {"fatigue_lambda", c.fatigueLambda},
             {"failure_exit_lambda", c.failureExitLambda}};
}

void from_json(const json &j, SessionDynamicsConfig &c) {
    ensureKnownKeys(
        j, "session_dynamics",
        {"topic_fatigue_weight", "creator_fatigue_weight", "novelty_match_weight",
         "topic_fatigue_increment", "creator_fatigue_increment", "general_fatigue_scale",
         "away_decay_half_life_seconds", "exit_bias", "exit_fatigue_weight", "exit_regret_weight",
         "exit_poor_streak_weight", "exit_satisfaction_weight", "exit_interruption_weight",
         "external_interruption_hazard", "regret_lambda", "fatigue_lambda", "failure_exit_lambda"});
    readKey(j, "topic_fatigue_weight", c.topicFatigueWeight);
    readKey(j, "creator_fatigue_weight", c.creatorFatigueWeight);
    readKey(j, "novelty_match_weight", c.noveltyMatchWeight);
    readKey(j, "topic_fatigue_increment", c.topicFatigueIncrement);
    readKey(j, "creator_fatigue_increment", c.creatorFatigueIncrement);
    readKey(j, "general_fatigue_scale", c.generalFatigueScale);
    readKey(j, "away_decay_half_life_seconds", c.awayDecayHalfLifeSeconds);
    readKey(j, "exit_bias", c.exitBias);
    readKey(j, "exit_fatigue_weight", c.exitFatigueWeight);
    readKey(j, "exit_regret_weight", c.exitRegretWeight);
    readKey(j, "exit_poor_streak_weight", c.exitPoorStreakWeight);
    readKey(j, "exit_satisfaction_weight", c.exitSatisfactionWeight);
    readKey(j, "exit_interruption_weight", c.exitInterruptionWeight);
    readKey(j, "external_interruption_hazard", c.externalInterruptionHazard);
    readKey(j, "regret_lambda", c.regretLambda);
    readKey(j, "fatigue_lambda", c.fatigueLambda);
    readKey(j, "failure_exit_lambda", c.failureExitLambda);
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
             {"behaviour_v2", c.behaviourV2},
             {"session_dynamics", c.sessionDynamics},
             {"reward", c.reward},
             {"evaluation", c.evaluation},
             {"realism", c.realism}};
}

void from_json(const json &j, ExperimentConfig &c) {
    ensureKnownKeys(j, "<top-level>",
                    {"simulation", "recommendation", "algorithm", "hnsw", "ranking", "learning",
                     "exploration", "diversity", "drift", "behaviour", "behaviour_v2",
                     "session_dynamics", "reward", "evaluation", "realism"});
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
    readKey(j, "behaviour_v2", c.behaviourV2);
    readKey(j, "session_dynamics", c.sessionDynamics);
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
