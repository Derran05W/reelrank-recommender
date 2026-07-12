#include "rr/evaluation/results_writer.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rr/infrastructure/config.hpp"

namespace rr {

namespace {

// Fixed-precision, classic-locale double formatting. The classic locale makes the output
// independent of the ambient LC_NUMERIC so the deterministic CSVs are byte-identical regardless of
// the environment (D8 / TDD 24.6).
std::string num(double v, int precision = 6) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

const char *kLearningEnabledNote =
    "Online preference learning is ENABLED (Phase 7, TDD 8.3/11.2/11.3): every user starts at the "
    "cold-start global-average estimate (TDD 11.1) and the OnlineUserStateUpdater updates the "
    "long-term/session/estimated preference vectors after each interaction. learning_curve.csv's "
    "mean_estimated_hidden_cosine should trend up as estimates converge toward the hidden "
    "preference.";

const char *kLearningFrozenNote =
    "Online preference learning is DISABLED (frozen arm): every user's estimatedPreference stays "
    "at "
    "the cold-start global-average hidden preference (TDD 11.1) with no online updates. Reported "
    "metrics reflect fixed, non-personalized-per-user estimates; learning_curve.csv's "
    "mean_estimated_hidden_cosine is constant across rounds.";

const char *kRegretUnitsNote =
    "Regret is measured in TRUE-AFFINITY units, not reward units: simulating counterfactual oracle "
    "interactions would consume behaviour rng draws and perturb determinism (D8), so the oracle "
    "compares mean true affinity of its exhaustive top-k against the recommended feed. Affinity is "
    "the monotone core of the reward (TDD 10.1/10.5).";

const char *kBaselineFlagsNote =
    "Requests set enableExploration = exploration.enabled (config-driven since Phase 8) and "
    "enableDiversity = diversity.enabled (config-driven since Phase 9). No existing recommender "
    "reads either flag, so both are inert for every non-exploration / non-diversity algorithm - "
    "the "
    "sibling orchestrator's gates are their only readers. candidateLimit = "
    "recommendation.vectorCandidates; requestTime = simulator logical clock; sessionId = the most "
    "recent interaction's session (SessionId{0} before any interaction).";

const char *kRetrievalNote =
    "Live retrieval quality (TDD 18.1) on a Bernoulli(retrieval_sample_rate) subset of requests: "
    "the recommender's ANN index is compared against an exact ground-truth index over the active "
    "reels for the same query, at k=50. Recall@K = |ANN_K intersect Exact_K| / min(K, corpus); "
    "Recall@10 uses the first 10 results. Distance error is the mean positionwise "
    "|d_ann,i - d_exact,i| over the first 10 results. Values are deterministic (exact searches). "
    "For an exact recommender this measures exactly recall 1.0 / distance error 0.0 (a wiring "
    "self-check).";

const char *kRetrievalNotApplicableNote =
    "The algorithm is not vector-based (retrievalIndex() == nullptr), so no live retrieval metrics "
    "are computed; retrieval_metrics.csv is still written with zero-sample rows for a uniform "
    "layout.";

const char *kColdStartNote =
    "Mid-simulation injection (Phase 8, TDD 18.5). Entities are injected at the START of their "
    "configured 0-based round; on a shared round REELS are injected before USERS. Injected users "
    "start from the run-START frozen cold-start prior (TDD 11.1) and receive requests from their "
    "injection round onward (their cold-start window). new_user_regret.first_N is the POOLED mean "
    "oracle regret over all injected users' first N impressions (-1 if none reached N); regret is "
    "in "
    "true-affinity units (see oracle note). target_reward is the run's pre-injection mean reward "
    "per "
    "impression (impressions in rounds before new_users_at); interactions_to_target_reward is the "
    "smallest 1-based impression count K at which injected users' cumulative mean reward reaches "
    "the "
    "target (-1 if undefined or not reached within the first 100 impressions). Curves are in "
    "new_user_curve.csv / new_reel_exposure.csv; all values are deterministic.";

const char *kDiversityNote =
    "Per-feed diversity (Phase 9, TDD 18.4), measured on EVERY request (unsampled: a feedSize-10 "
    "feed is 45 dot products) from the feed AS PRESENTED, before its impressions are stepped, so "
    "the seen-set is the pre-feed state. unique_topics/creators = distinct primaryTopic/creatorId "
    "counts; intra_list_similarity = mean pairwise embedding cosine (0 for feeds with <2 items); "
    "topic_hhi/creator_hhi = Herfindahl-Hirschman index over topic/creator shares (sum of squared "
    "shares: 1.0 = one topic, 1/k = uniform over k distinct). repetition_rate = fraction of feed "
    "items that were shown to the user earlier this run OR duplicated within the feed - expected 0 "
    "by construction (the orchestrator seen-filters and de-duplicates), published as live "
    "verification of the 'duplicate/repetitive content eliminated' exit criterion. The per-round "
    "curve is in diversity_metrics.csv; all values are deterministic. This is measured for ANY "
    "algorithm (diversity.enabled does not change existing feeds), so the numbers are the "
    "phase-comparison baseline.";

// p50/p95/p99/mean/max/samples of a LatencyStats as a JSON object (wall-clock, D9).
nlohmann::json latencyJson(const LatencyStats &l) {
    return nlohmann::json{{"p50", l.p50Ms},   {"p95", l.p95Ms}, {"p99", l.p99Ms},
                          {"mean", l.meanMs}, {"max", l.maxMs}, {"samples", l.samples}};
}

} // namespace

void ResultsWriter::writeConfigJson(const ExperimentResult &result) {
    // Fully-resolved config, written back out (D6). nlohmann orders object keys, so this file is
    // byte-identical across runs with the same config.
    const nlohmann::json j = result.config;
    std::ofstream out(result.directory / "config.json");
    out << j.dump(2) << "\n";
}

void ResultsWriter::writeSummaryJson(const ExperimentResult &result) {
    const MetricsSummary &m = result.overall;
    nlohmann::json j;
    j["experiment_id"] = result.experimentId;
    j["algorithm"] = toString(result.config.algorithm);
    j["seed"] = result.seed;

    j["counts"] = {{"users", result.userCount},       {"reels", result.reelCount},
                   {"requests", result.requestCount}, {"impressions", result.impressionCount},
                   {"sessions", m.sessions},          {"rounds", result.rounds.size()}};

    j["metrics"] = {{"mean_watch_ratio", m.meanWatchRatio},
                    {"mean_watch_seconds", m.meanWatchSeconds},
                    {"instant_skip_rate", m.instantSkipRate},
                    {"completion_rate", m.completionRate},
                    {"like_rate", m.likeRate},
                    {"share_rate", m.shareRate},
                    {"follow_rate", m.followRate},
                    {"mean_session_length", m.meanSessionLength},
                    {"reward_per_impression", m.rewardPerImpression},
                    {"reward_per_session", m.rewardPerSession},
                    {"mean_true_affinity", m.meanTrueAffinity}};

    j["oracle"] = {{"sample_rate", result.oracleSampleRate},
                   {"sampled_requests", result.sampledRequestCount},
                   {"mean_regret", result.meanRegret},
                   {"cumulative_regret", result.cumulativeRegret},
                   {"regret_units_note", kRegretUnitsNote}};

    // Online preference learning (Phase 7). `final_estimated_hidden_cosine` is the mean
    // cos(estimatedPreference, hiddenPreference) over all users at the end of the last round
    // (TDD 18.5) - the headline convergence number; deterministic.
    j["learning"] = {{"enabled", result.learningEnabled},
                     {"final_estimated_hidden_cosine", result.finalEstimatedHiddenCosine},
                     {"note", result.learningEnabled ? kLearningEnabledNote : kLearningFrozenNote}};

    // Live retrieval quality (TDD 18.1). Deterministic (exact index searches), unlike the timing
    // subsection below. `note` explains whether metrics were computed for this algorithm.
    j["retrieval"] = {
        {"applicable", result.retrievalApplicable},
        {"sample_rate", result.retrievalSampleRate},
        {"sampled_requests", result.retrievalSampleCount},
        {"recall_at_10", result.retrievalRecallAt10},
        {"recall_at_50", result.retrievalRecallAt50},
        {"mean_distance_error", result.retrievalDistanceError},
        {"note", result.retrievalApplicable ? kRetrievalNote : kRetrievalNotApplicableNote}};

    // Diversity block (Phase 9, TDD 18.4): overall means over every feed + the run's total repeat
    // count. UNCONDITIONAL - present for every run (diversity is measurable for any algorithm), so
    // the phase comparison has baseline numbers. Deterministic (rng/clock-free). `repetition_total`
    // is expected 0 by construction (note explains).
    j["diversity"] = {{"feeds", result.diversityFeedCount},
                      {"mean_unique_topics", result.meanUniqueTopics},
                      {"mean_unique_creators", result.meanUniqueCreators},
                      {"mean_intra_list_similarity", result.meanIntraListSimilarity},
                      {"mean_topic_hhi", result.meanTopicConcentration},
                      {"mean_creator_hhi", result.meanCreatorConcentration},
                      {"repetition_total", result.totalRepetitions},
                      {"repetition_rate", result.repetitionRate},
                      {"note", kDiversityNote}};

    // Cold-start / injection block (Phase 8, TDD 18.5): PRESENT only when injection is configured,
    // so a non-configured run's summary.json carries no `cold_start` key (byte-identical to a
    // pre-Phase-8 run's non-timing content).
    if (result.coldStart.configured) {
        const ColdStartReport &c = result.coldStart;
        j["cold_start"] = {{"new_users", c.newUsers},
                           {"new_users_at", c.newUsersAt},
                           {"new_reels", c.newReels},
                           {"new_reels_at", c.newReelsAt},
                           {"new_user_regret",
                            {{"first_10", c.meanRegretFirst10},
                             {"first_25", c.meanRegretFirst25},
                             {"first_50", c.meanRegretFirst50},
                             {"first_100", c.meanRegretFirst100}}},
                           {"target_reward_defined", c.targetDefined},
                           {"target_reward", c.targetReward},
                           {"interactions_to_target_reward", c.interactionsToTargetReward},
                           {"new_reel_exposure",
                            {{"total_injected_impressions", c.totalInjectedImpressions},
                             {"distinct_injected_exposed", c.distinctInjectedExposed},
                             {"share_of_all_impressions", c.injectedImpressionShare}}},
                           {"note", kColdStartNote}};
    }

    j["notes"] = {{"learning", result.learningEnabled ? kLearningEnabledNote : kLearningFrozenNote},
                  {"baseline_flags", kBaselineFlagsNote}};

    // Wall-clock timing is confined to this subsection + latency_metrics.csv + metadata.json; it is
    // intentionally NOT part of the determinism guarantee (D9). The per-stage stats decompose the
    // recommend() call (TDD 18.7); they are all-zero for recommenders that do not populate the
    // per-stage response fields (Random/Popularity).
    j["timing"] = {{"total_wall_seconds", result.totalWallSeconds},
                   {"recommend_latency_ms", latencyJson(result.latency)},
                   {"retrieval_latency_ms", latencyJson(result.retrievalLatency)},
                   {"ranking_latency_ms", latencyJson(result.rankingLatency)},
                   {"reranking_latency_ms", latencyJson(result.rerankingLatency)}};

    std::ofstream out(result.directory / "summary.json");
    out << j.dump(2) << "\n";
}

void ResultsWriter::writeRetrievalMetricsCsv(const ExperimentResult &result) {
    // Deterministic (exact index searches): part of the byte-identical determinism guarantee.
    // 0-sample rounds (and every round of a non-vector algorithm) write zeros for a uniform layout.
    std::ofstream csv(result.directory / "retrieval_metrics.csv");
    csv << "round,samples,recall_at_10,recall_at_50,mean_distance_error\n";
    for (const RoundMetrics &r : result.rounds) {
        csv << r.round << ',' << r.retrievalSamples << ',' << num(r.meanRecallAt10) << ','
            << num(r.meanRecallAt50) << ',' << num(r.meanDistanceError) << '\n';
    }
}

void ResultsWriter::writeRecommendationMetricsCsv(const ExperimentResult &result) {
    std::ofstream csv(result.directory / "recommendation_metrics.csv");
    csv << "round,impressions,mean_watch_ratio,mean_watch_seconds,instant_skip_rate,"
           "completion_rate,like_rate,share_rate,follow_rate,mean_session_length,"
           "reward_per_impression,reward_per_session,mean_true_affinity\n";
    for (const RoundMetrics &r : result.rounds) {
        const MetricsSummary &m = r.metrics;
        csv << r.round << ',' << m.impressions << ',' << num(m.meanWatchRatio) << ','
            << num(m.meanWatchSeconds) << ',' << num(m.instantSkipRate) << ','
            << num(m.completionRate) << ',' << num(m.likeRate) << ',' << num(m.shareRate) << ','
            << num(m.followRate) << ',' << num(m.meanSessionLength) << ','
            << num(m.rewardPerImpression) << ',' << num(m.rewardPerSession) << ','
            << num(m.meanTrueAffinity) << '\n';
    }
}

void ResultsWriter::writeDiversityMetricsCsv(const ExperimentResult &result) {
    // Per-feed diversity (Phase 9, TDD 18.4), one row per round: the means over that round's feeds
    // plus the round's repetition rate (repeats / total feed items). Deterministic (num() fixed
    // precision, classic locale): byte-identical across same-seed runs and part of the determinism
    // guarantee. Written UNCONDITIONALLY (unlike the Phase 8 injection files) - diversity is
    // measurable for any algorithm and is the phase-comparison baseline. repetition_rate is
    // expected 0 by construction (see the summary.json diversity note).
    std::ofstream csv(result.directory / "diversity_metrics.csv");
    csv << "round,mean_unique_topics,mean_unique_creators,mean_intra_list_similarity,"
           "mean_topic_hhi,mean_creator_hhi,repetition_rate\n";
    for (const RoundMetrics &r : result.rounds) {
        csv << r.round << ',' << num(r.meanUniqueTopics) << ',' << num(r.meanUniqueCreators) << ','
            << num(r.meanIntraListSimilarity) << ',' << num(r.meanTopicConcentration) << ','
            << num(r.meanCreatorConcentration) << ',' << num(r.repetitionRate) << '\n';
    }
}

void ResultsWriter::writeLearningCurveCsv(const ExperimentResult &result) {
    // Cold-start learning curve (Phase 7, TDD 18.5): reward and estimate<->hidden alignment against
    // cumulative interaction count. `interactions_per_user` is the per-user interaction budget
    // spent through this round: min((round+1)*feedSize, interactionsPerUser). With learning enabled
    // mean_estimated_hidden_cosine trends up as estimates converge; with learning disabled (frozen
    // arm) it is CONSTANT across rounds. Deterministic (num() fixed precision): byte-identical
    // across same-seed runs.
    const size_t feedSize = result.config.recommendation.feedSize;
    const size_t interactionsPerUser = result.config.simulation.interactionsPerUser;
    std::ofstream csv(result.directory / "learning_curve.csv");
    csv << "round,interactions_per_user,mean_reward_per_impression,mean_estimated_hidden_cosine\n";
    for (const RoundMetrics &r : result.rounds) {
        const size_t interactions = std::min((r.round + 1) * feedSize, interactionsPerUser);
        csv << r.round << ',' << interactions << ',' << num(r.metrics.rewardPerImpression) << ','
            << num(r.meanEstimatedHiddenCosine) << '\n';
    }
}

void ResultsWriter::writeRegretCurveCsv(const ExperimentResult &result) {
    std::ofstream csv(result.directory / "regret_curve.csv");
    csv << "round,sampled_requests,mean_regret,cumulative_regret\n";
    for (const RoundMetrics &r : result.rounds) {
        csv << r.round << ',' << r.sampledRequests << ',' << num(r.meanRegret) << ','
            << num(r.cumulativeRegret) << '\n';
    }
}

void ResultsWriter::writeLatencyMetricsCsv(const ExperimentResult &result) {
    // Wall-clock file: NOT part of the determinism guarantee. Long format, one row per stage: the
    // whole recommend() call ("total") plus its retrieval/ranking/reranking decomposition (TDD 18.7
    // / Phase 5 stage-percentile exit criterion). Stage rows are all-zero for recommenders that do
    // not populate the per-stage response fields (Random/Popularity).
    std::ofstream csv(result.directory / "latency_metrics.csv");
    csv << "stage,p50_ms,p95_ms,p99_ms,mean_ms,max_ms,num_samples\n";
    const auto row = [&csv](const char *stage, const LatencyStats &l) {
        csv << stage << ',' << num(l.p50Ms) << ',' << num(l.p95Ms) << ',' << num(l.p99Ms) << ','
            << num(l.meanMs) << ',' << num(l.maxMs) << ',' << l.samples << '\n';
    };
    row("total", result.latency);
    row("retrieval", result.retrievalLatency);
    row("ranking", result.rankingLatency);
    row("reranking", result.rerankingLatency);
}

void ResultsWriter::writeNewUserCurveCsv(const ExperimentResult &result) {
    // New-user cold-start curve (Phase 8, TDD 18.5): per per-user impression index, the pooled mean
    // reward and mean oracle regret over INJECTED users. Deterministic (num() fixed precision);
    // header-only when no injected user reached any tracked impression index.
    std::ofstream csv(result.directory / "new_user_curve.csv");
    csv << "impression_index,users_at_index,mean_reward,mean_regret\n";
    for (const NewUserCurvePoint &p : result.coldStart.newUserCurve) {
        csv << p.impressionIndex << ',' << p.usersAtIndex << ',' << num(p.meanReward) << ','
            << num(p.meanRegret) << '\n';
    }
}

void ResultsWriter::writeNewReelExposureCsv(const ExperimentResult &result) {
    // New-reel exposure (Phase 8, TDD 18.5): per round, impressions on injected reels with running
    // totals, cumulative distinct injected reels exposed, and this round's share of all
    // impressions. Deterministic. One row per round.
    std::ofstream csv(result.directory / "new_reel_exposure.csv");
    csv << "round,injected_impressions,injected_impressions_cum,distinct_injected_exposed_cum,"
           "share_of_round_impressions\n";
    for (const NewReelExposurePoint &p : result.coldStart.newReelExposure) {
        csv << p.round << ',' << p.injectedImpressions << ',' << p.injectedImpressionsCum << ','
            << p.distinctInjectedExposedCum << ',' << num(p.shareOfRoundImpressions) << '\n';
    }
}

void ResultsWriter::writeAll(const ExperimentResult &result, const RunMetadata &meta) {
    writeConfigJson(result);
    writeSummaryJson(result);
    writeRetrievalMetricsCsv(result);
    writeRecommendationMetricsCsv(result);
    // Phase 9: UNCONDITIONAL (diversity is measurable for any algorithm) - adds one file for every
    // run without perturbing any pre-existing file.
    writeDiversityMetricsCsv(result);
    writeLearningCurveCsv(result);
    writeRegretCurveCsv(result);
    writeLatencyMetricsCsv(result);
    // Phase 8 injection files: written ONLY when injection is configured, so a normal run's output
    // directory is byte-for-byte a pre-Phase-8 run's (no extra files).
    if (result.coldStart.configured) {
        writeNewUserCurveCsv(result);
        writeNewReelExposureCsv(result);
    }
    writeMetadataJson(result.directory, meta);
}

} // namespace rr
