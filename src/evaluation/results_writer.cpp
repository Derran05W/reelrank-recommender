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

const char *kAdaptationNote =
    "Preference-drift adaptation (Phase 10, TDD 18.6). Drift is scheduled per user cohort: when a "
    "cohort user reaches atInteraction completed interactions, the simulator rebuilds their HIDDEN "
    "preference from the event's topic mix (recommender-visible state is untouched at the drift "
    "instant). The 'drifted' cohort = users in >= 1 event's cohort; 'control' = the rest. "
    "drift_round = first_drift_interaction / feed_size (integer floor): the first round whose "
    "feeds "
    "can be affected, so rounds < drift_round are strictly pre-drift. pre_drift_reward = mean of "
    "the "
    "drifted cohort's per-round mean reward over up to the 3 rounds before drift_round (-1 if "
    "drift_round == 0). trough_reward/round = min drifted-cohort round reward at/after "
    "drift_round; "
    "reward_drop = pre_drift_reward - trough_reward. recovery_round = first round >= drift_round "
    "with "
    "drifted reward >= 0.95*pre_drift_reward (recovery_interactions = (recovery_round - "
    "drift_round + "
    "1)*feed_size; both -1 if never/undefined). pre_drift_alignment = drifted-cohort est<->hidden "
    "cosine at round drift_round-1; alignment_recovery_round = first round >= drift_round with "
    "drifted "
    "alignment >= 0.95*pre_drift_alignment -- the 'interactions until new preference detection' "
    "reading. adaptation_window_regret = sum of the run's per-round SAMPLED oracle regret over "
    "[drift_round, recovery_round|last] (whole-population, true-affinity units; see the regret "
    "note). "
    "learning_curve.csv gains four columns (drifted_mean_reward, control_mean_reward, "
    "drifted_alignment, control_alignment); an empty cohort prints 'nan'. All values "
    "deterministic.";

const char *kWelfareNote =
    "Hidden user welfare group (Phase 15, V2 TDD §6). Present only under realism.latent_reactions. "
    "Every impression computes a hidden LatentReaction whose immediate satisfaction ([-1,1]) and "
    "regret ([0,1]) are aggregated here through the D18 EVALUATION CARVE-OUT — the latent never "
    "reaches any recommender-visible structure (no latent field is ever serialized on an "
    "InteractionEvent or User; the leak audit enforces this). These are ground-truth welfare, "
    "DISTINCT from the engagement-proxy reward the online learner optimizes: engagement is "
    "evidence, not truth (V2 TDD 3.2), so mean_immediate_satisfaction can diverge from "
    "reward_per_impression. satisfaction_per_minute = sum(immediate satisfaction) / sum(simulated "
    "watch-minutes) [watch_seconds/60] — satisfaction accrued per minute of watch time (can be "
    "negative), NOT a per-impression mean; watch_minutes is the reproducible denominator. "
    "archetype_exposure gives each hidden archetype's impression share (the direct 'does the "
    "engagement arm over-serve ragebait/clickbait?' measurement) plus its mean "
    "satisfaction/regret, "
    "one row per catalog archetype in index order. harmful_fatigue and platform_trust are "
    "NOT-YET-MODELED placeholders (constant 0) — harmful fatigue becomes real in P16 (session "
    "dynamics), platform trust in P20 (retention); emitted now so the schema is stable early "
    "(D22). "
    "Per-round rows are in welfare_metrics.csv, per-archetype rows in "
    "welfare_archetype_metrics.csv. "
    "Deterministic.";

const char *kMetricGroupsNote =
    "V2 TDD §6 defines FOUR metric groups reported as SEPARATE blocks; NO single aggregate score "
    "is "
    "ever defined (D22) — a policy may improve one group while damaging another. (1) engagement: "
    "the V1 metrics{} block (watch/completion/like/share/follow) + recommendation_metrics.csv, "
    "extended here with the V2 comment/save/profile-visit rates. (2) hidden_user_welfare: the "
    "welfare{} block + welfare_metrics.csv + welfare_archetype_metrics.csv (this run's welfare "
    "group). (3) session_health: LIMITED before Phase 16 — only session length / "
    "reward-per-session "
    "exist so far; probabilistic classified exits, early-failure-exit rate, return delay and "
    "retention arrive in P16/P20. (4) recommendation_quality: the diversity{} block + "
    "metrics.mean_true_affinity + learning.final_estimated_hidden_cosine (V1 §18.2/18.4). The "
    "engagement additions in this block are the ONLY new engagement numbers; the existing V1 "
    "engagement files/columns are unchanged (D22).";

const char *kSessionHealthNote =
    "Session-health group (Phase 16, V2 TDD §4.9/§6). Present only under realism.session_dynamics "
    "(which requires latent_reactions). Session length is an OUTCOME of feed quality: the harness "
    "consumes each feed impression-by-impression and STOPS when a probabilistic classified exit "
    "fires (V2 TDD 4.8), collecting one COMPLETED SessionRecord per exit through the D18 "
    "EVALUATION "
    "CARVE-OUT (SessionRecord's hidden-derived satisfaction/regret/harmful-fatigue never reach a "
    "recommender-visible structure). All means/rates here are over CLOSED sessions; a session "
    "still "
    "open when the simulation ends is a RunEnded record, counted in open_sessions and EXCLUDED "
    "from "
    "every mean/rate/share denominator (it is not a real exit). satisfaction_per_minute / "
    "regret_per_minute divide by SESSION-DURATION minutes (time before exit / 60) — deliberately "
    "distinct from the hidden-welfare group's WATCH-minute denominator. mean_session_utility is "
    "U_s = Σsatisfaction − λ1·Σregret − λ2·harmfulFatigue − λ3·[failure exit] (λs from "
    "session_dynamics config), recomputed on the eval side from the record components. "
    "early_failure_exit_rate / natural_completion_rate are the Failure / Satisfied shares of "
    "closed "
    "sessions; exit_type_shares gives the full V2 §4.8 taxonomy distribution. "
    "next_session_starting_satisfaction is the §4.9 measure: mean carry-over satisfaction of "
    "sessions that followed an earlier session of the same user (consecutive records linked per "
    "user). harmful_fatigue_mean also realizes the hidden-welfare group's previously-placeholder "
    "harmful_fatigue column under this gate. Per-round rows are in session_health.csv; all values "
    "are deterministic. NOTE (integrator): the scaffold provides no run-end drain hook, so open "
    "sessions are not yet collected as RunEnded records — open_sessions is 0 until package A's "
    "exit "
    "model / a drain hook lands; the reduction already handles RunEnded correctly.";

const char *kEventModeNote =
    "Event-mode additions (Phase 18, V2 TDD §4.11/4.12/4.14, D20). Present ONLY under "
    "simulation.scheduler='event_queue'; a round-robin run's summary.json carries no event_mode "
    "key "
    "(byte-identical legacy path, D17). Users open/scroll/exit/return on INDEPENDENT timelines "
    "over "
    "a deterministic (time, pinned-tie-breaker) priority queue, so per-user interaction counts are "
    "an OUTCOME (interactions_per_user is ignored). event_log_digest is a pure SplitMix64 fold "
    "over "
    "the full event stream in processing order (the D20 'same seed => identical event sequence' "
    "tripwire; unchanged under user-init/insertion-order permutation). The four §6 groups above "
    "are "
    "reported per SIMULATED DAY (86400s). sessions_per_simulated_day = closed sessions / "
    "simulated_days; mean_concurrent_online is the mean fraction of users online sampled once per "
    "processed event timestamp; return_delay stats are the baseline model's per-exit draws "
    "(max(60, gaussian(mean/baselineDailyUsage, spread)) seconds, stream 'scheduling').";

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

    // Preference-drift adaptation block (Phase 10, TDD 18.6): PRESENT only when drift is
    // configured, so a non-drift run's summary.json carries no `adaptation` key (byte-identical to
    // a pre-Phase-10 run's non-timing content). The plotting package depends on the exact key names
    // `first_drift_interaction` and `pre_drift_reward`; the rest are documented in kAdaptationNote.
    if (result.adaptation.configured) {
        const AdaptationReport &a = result.adaptation;
        nlohmann::json events = nlohmann::json::array();
        for (const DriftEvent &e : result.config.drift.events) {
            nlohmann::json mix = nlohmann::json::array();
            for (const DriftTopicWeight &w : e.topicMix) {
                mix.push_back({{"topic", w.topic}, {"weight", w.weight}});
            }
            events.push_back({{"at_interaction", e.atInteraction},
                              {"cohort_lo", e.cohortLo},
                              {"cohort_hi", e.cohortHi},
                              {"topic_mix", mix}});
        }
        j["adaptation"] = {{"events", events},
                           {"drifted_users", a.driftedUsers},
                           {"control_users", a.controlUsers},
                           {"first_drift_interaction", a.firstDriftInteraction},
                           {"drift_round", a.driftRound},
                           {"feed_size", a.feedSize},
                           {"pre_drift_reward", a.preDriftReward},
                           {"trough_reward", a.troughReward},
                           {"trough_round", a.troughRound},
                           {"reward_drop", a.rewardDrop},
                           {"recovery_round", a.recoveryRound},
                           {"recovery_interactions", a.recoveryInteractions},
                           {"pre_drift_alignment", a.preDriftAlignment},
                           {"post_drift_alignment_min", a.postDriftAlignmentMin},
                           {"post_drift_alignment_min_round", a.postDriftAlignmentMinRound},
                           {"alignment_recovery_round", a.alignmentRecoveryRound},
                           {"adaptation_window_regret", a.adaptationWindowRegret},
                           {"note", kAdaptationNote}};
    }

    // Hidden-user-welfare block (Phase 15, V2 TDD §6, D22): PRESENT only under
    // realism.latent_reactions, so a gate-off run's summary.json carries no `welfare` key
    // (byte-identical to a pre-Phase-14 run's non-timing content, D17). The first three value keys
    // are unchanged from Phase 14; Phase 15 appends satisfaction_per_minute, watch_minutes, the two
    // NOT-YET-MODELED placeholders, and the per-archetype exposure array (overall, index order).
    if (result.welfare.configured) {
        const WelfareReport &w = result.welfare;
        nlohmann::json exposure = nlohmann::json::array();
        for (const ArchetypeWelfare &a : w.byArchetype) {
            exposure.push_back({{"archetype_index", a.archetypeIndex},
                                {"archetype_name", a.name},
                                {"impressions", a.impressions},
                                {"exposure_share", a.exposureShare},
                                {"mean_immediate_satisfaction", a.meanSatisfaction},
                                {"mean_regret", a.meanRegret}});
        }
        // harmful_fatigue is a NOT-YET-MODELED placeholder UNTIL Phase 16: under
        // realism.session_dynamics it is realized from session data (the mean end-of-session
        // harmful fatigue, filled by the runner from the session-health report), so it drops out of
        // not_yet_modeled and gains a source note. Gated on sessionHealth.configured so a P15-only
        // run (latent_reactions on, session_dynamics off) is byte-identical to before (D17).
        const bool harmfulFatigueModeled = result.sessionHealth.configured;
        const nlohmann::json notYetModeled =
            harmfulFatigueModeled ? nlohmann::json::array({"platform_trust"})
                                  : nlohmann::json::array({"harmful_fatigue", "platform_trust"});
        j["welfare"] = {{"impressions", w.impressions},
                        {"mean_immediate_satisfaction", w.meanSatisfaction},
                        {"mean_regret", w.meanRegret},
                        {"satisfaction_per_minute", w.satisfactionPerMinute},
                        {"watch_minutes", w.watchMinutes},
                        {"harmful_fatigue", w.harmfulFatigue},
                        {"platform_trust", w.platformTrust},
                        {"not_yet_modeled", notYetModeled},
                        {"archetype_exposure", exposure},
                        {"note", kWelfareNote}};
        if (harmfulFatigueModeled) {
            j["welfare"]["harmful_fatigue_source"] =
                "Phase 16 (realism.session_dynamics): mean end-of-session harmful fatigue over "
                "closed sessions (see the session_health block / session_health.csv) — no longer a "
                "placeholder under this gate.";
        }

        // Four-group index (V2 §6, D22): a single gate-on block documenting the four groups and
        // carrying the engagement group's V2 additions (comment/save/profile-visit rates from the
        // overall MetricsSummary). recommendation-quality points at the existing V1 fields. The
        // session-health group is LIMITED pre-P16 (V1 session-length only); under
        // realism.session_dynamics it becomes LIVE and points at the session_health block/CSV. The
        // else-branch is byte-for-byte the pre-Phase-16 object so a P15-only run is unchanged
        // (D17). NO aggregate score anywhere.
        const MetricsSummary &ov = result.overall;
        nlohmann::json sessionHealthGroup;
        if (result.sessionHealth.configured) {
            sessionHealthGroup = {
                {"status", "live"},
                {"see", "session_health{} block + session_health.csv"},
                {"available",
                 {"session_count", "time_before_exit_mean", "time_before_exit_median",
                  "mean_impressions_per_session", "satisfaction_per_minute", "regret_per_minute",
                  "mean_session_utility", "early_failure_exit_rate", "natural_completion_rate",
                  "next_session_starting_satisfaction", "exit_type_shares",
                  "harmful_fatigue_mean"}},
                {"pending", {"return_delay", "sessions_per_day", "retention"}}};
        } else {
            sessionHealthGroup = {
                {"status", "limited_pre_p16"},
                {"available", {"mean_session_length", "reward_per_session"}},
                {"pending",
                 {"early_failure_exit_rate", "natural_exit_rate", "return_delay",
                  "sessions_per_day", "retention"}},
                {"note",
                 "Session-health group is LIMITED before Phase 16: only the V1 session-length "
                 "/ reward-per-session numbers exist. Probabilistic classified exits, "
                 "early-failure-exit rate, return delay and retention arrive in P16/P20."}};
        }
        j["metric_groups"] = {
            {"note", kMetricGroupsNote},
            {"engagement",
             {{"v1_fields", "metrics{} block + recommendation_metrics.csv "
                            "(watch/completion/like/share/follow)"},
              {"comment_rate", ov.commentRate},
              {"save_rate", ov.saveRate},
              {"profile_visit_rate", ov.profileVisitRate}}},
            {"hidden_user_welfare",
             {{"see", "welfare{} block + welfare_metrics.csv + welfare_archetype_metrics.csv"}}},
            {"session_health", sessionHealthGroup},
            {"recommendation_quality",
             {{"see", "diversity{} block + metrics.mean_true_affinity + "
                      "learning.final_estimated_hidden_cosine"}}}};
    }

    // Session-health block (Phase 16, V2 TDD §4.9/§6, D22): PRESENT only under
    // realism.session_dynamics, so a P15-only or gate-off run's summary.json carries no
    // `session_health` key (byte-identical to a pre-Phase-16 run's non-timing content, D17).
    // Overall session statistics + the exit-type distribution (counts + shares); RunEnded is
    // reported as open_sessions and excluded from every share/rate denominator (see the note).
    if (result.sessionHealth.configured) {
        const SessionHealthReport &s = result.sessionHealth;
        j["session_health"] = {
            {"sessions", s.sessions},
            {"open_sessions", s.openSessions},
            {"mean_duration_seconds", s.meanDurationSeconds},
            {"median_duration_seconds", s.medianDurationSeconds},
            {"mean_impressions_per_session", s.meanImpressions},
            {"duration_minutes", s.durationMinutes},
            {"satisfaction_per_minute", s.satisfactionPerMinute},
            {"regret_per_minute", s.regretPerMinute},
            {"mean_session_utility", s.meanSessionUtility},
            {"early_failure_exit_rate", s.earlyFailureExitRate},
            {"natural_completion_rate", s.naturalCompletionRate},
            {"harmful_fatigue_mean", s.harmfulFatigueMean},
            {"next_session_starting_satisfaction", s.nextSessionStartingSatisfaction},
            {"linked_sessions", s.linkedSessions},
            {"exit_type_counts",
             {{"failure", s.exits.failure},
              {"satisfied", s.exits.satisfied},
              {"fatigue", s.exits.fatigue},
              {"external", s.exits.external},
              {"regret", s.exits.regret},
              {"run_ended", s.exits.runEnded}}},
            {"exit_type_shares",
             {{"failure", s.exitShare(SessionExitType::Failure)},
              {"satisfied", s.exitShare(SessionExitType::Satisfied)},
              {"fatigue", s.exitShare(SessionExitType::Fatigue)},
              {"external", s.exitShare(SessionExitType::External)},
              {"regret", s.exitShare(SessionExitType::Regret)},
              {"open", s.openShare()}}},
            {"note", kSessionHealthNote}};
    }

    // Event-mode block (Phase 18, V2 TDD §4.11/4.12/4.14, D20/D22): PRESENT only under the
    // event-queue scheduler, so a round-robin run's summary.json carries no `event_mode` key
    // (byte-identical to the legacy path, D17). Carries the deterministic event-log digest + count
    // (the D20 tripwire package C golden-tests) and the event-mode session-health additions the
    // round-robin loop cannot produce (sessions/simulated-day, concurrent-online occupancy, return
    // delays). Purely additive — appended after every existing block so nothing shifts.
    if (result.eventMode.configured) {
        const EventModeReport &em = result.eventMode;
        j["event_mode"] = {{"scheduler", "event_queue"},
                           {"event_count", em.eventCount},
                           {"event_log_digest", em.eventLogDigest},
                           {"simulated_days", em.simulatedDays},
                           {"sessions_per_simulated_day", em.sessionsPerSimulatedDay},
                           {"mean_concurrent_online", em.meanConcurrentOnline},
                           {"return_delay_seconds",
                            {{"mean", em.returnDelayMeanSeconds},
                             {"median", em.returnDelayMedianSeconds},
                             {"count", em.returnCount}}},
                           {"note", kEventModeNote}};
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
    // Phase 10 (TDD 18.6): when drift is configured, append the drifted/control cohort split as
    // FOUR extra columns AFTER the legacy four. When NOT configured the format is byte-identical to
    // a pre-Phase-10 run (the regression contract). An empty cohort (e.g. control under
    // whole-population drift) prints the literal `nan` rather than a meaningless 0.0.
    const bool drift = result.adaptation.configured;
    std::ofstream csv(result.directory / "learning_curve.csv");
    csv << "round,interactions_per_user,mean_reward_per_impression,mean_estimated_hidden_cosine";
    if (drift) {
        csv << ",drifted_mean_reward,control_mean_reward,drifted_alignment,control_alignment";
    }
    csv << '\n';
    for (const RoundMetrics &r : result.rounds) {
        const size_t interactions = std::min((r.round + 1) * feedSize, interactionsPerUser);
        csv << r.round << ',' << interactions << ',' << num(r.metrics.rewardPerImpression) << ','
            << num(r.meanEstimatedHiddenCosine);
        if (drift) {
            csv << ',' << (r.driftedImpressions > 0 ? num(r.driftedMeanReward) : "nan") << ','
                << (r.controlImpressions > 0 ? num(r.controlMeanReward) : "nan") << ','
                << (r.driftedAlignUsers > 0 ? num(r.driftedAlignment) : "nan") << ','
                << (r.controlAlignUsers > 0 ? num(r.controlAlignment) : "nan");
        }
        csv << '\n';
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

void ResultsWriter::writeWelfareMetricsCsv(const ExperimentResult &result) {
    // Hidden-user-welfare group, per round (Phase 15, V2 TDD §6). Written ONLY under
    // realism.latent_reactions. The welfare columns come from result.welfare.byRound; the three V2
    // engagement rates come from the SAME round's engagement group (result.rounds[r].metrics) so
    // the engagement additions live only here, never in a V1 CSV (D22).
    // harmful_fatigue/platform_trust are NOT-YET-MODELED placeholders (constant 0; real in
    // P16/P20). Deterministic (num() fixed precision, classic locale): byte-identical across
    // same-seed runs.
    std::ofstream csv(result.directory / "welfare_metrics.csv");
    csv << "round,impressions,mean_immediate_satisfaction,mean_regret,satisfaction_per_minute,"
           "watch_minutes,comment_rate,save_rate,profile_visit_rate,harmful_fatigue,platform_"
           "trust\n";
    for (std::size_t r = 0; r < result.welfare.byRound.size(); ++r) {
        const WelfareRoundPoint &p = result.welfare.byRound[r];
        // Engagement rates for the same round (0 if the round index is somehow absent).
        const MetricsSummary eng =
            r < result.rounds.size() ? result.rounds[r].metrics : MetricsSummary{};
        csv << p.round << ',' << p.impressions << ',' << num(p.meanSatisfaction) << ','
            << num(p.meanRegret) << ',' << num(p.satisfactionPerMinute) << ','
            << num(p.watchMinutes) << ',' << num(eng.commentRate) << ',' << num(eng.saveRate) << ','
            << num(eng.profileVisitRate) << ',' << num(p.harmfulFatigue) << ','
            << num(p.platformTrust) << '\n';
    }
}

void ResultsWriter::writeWelfareArchetypeMetricsCsv(const ExperimentResult &result) {
    // Per-archetype exposure + welfare over the whole run (Phase 15, V2 TDD §6). Written ONLY under
    // realism.latent_reactions. One row per catalog archetype in index order (stable schema across
    // arms, even for zero-impression archetypes) — exposure_share is the direct measurement task
    // 5's engagement-vs-satisfaction experiment reads. Deterministic.
    std::ofstream csv(result.directory / "welfare_archetype_metrics.csv");
    csv << "archetype_index,archetype_name,impressions,exposure_share,mean_immediate_satisfaction,"
           "mean_regret\n";
    for (const ArchetypeWelfare &a : result.welfare.byArchetype) {
        csv << a.archetypeIndex << ',' << a.name << ',' << a.impressions << ','
            << num(a.exposureShare) << ',' << num(a.meanSatisfaction) << ',' << num(a.meanRegret)
            << '\n';
    }
}

void ResultsWriter::writeSessionHealthMetricsCsv(const ExperimentResult &result) {
    // Session-health group, per round (Phase 16, V2 TDD §4.9/§6). Written ONLY under
    // realism.session_dynamics. One row per round: the round's CLOSED-session statistics + U_s +
    // exit-type counts. All means/rates are over CLOSED sessions; RunEnded (open) sessions appear
    // only in open_sessions. Deterministic (num() fixed precision, classic locale): byte-identical
    // across same-seed runs. Under the P16 scaffold stub the loop closes zero sessions, so every
    // row is a well-formed zero row.
    std::ofstream csv(result.directory / "session_health.csv");
    csv << "round,sessions,open_sessions,mean_duration_seconds,median_duration_seconds,"
           "mean_impressions,satisfaction_per_minute,regret_per_minute,mean_session_utility,"
           "early_failure_exit_rate,natural_completion_rate,harmful_fatigue_mean,"
           "next_session_starting_satisfaction,failure_exits,satisfied_exits,fatigue_exits,"
           "external_exits,regret_exits\n";
    for (const SessionHealthRoundPoint &p : result.sessionHealth.byRound) {
        csv << p.round << ',' << p.sessions << ',' << p.openSessions << ','
            << num(p.meanDurationSeconds) << ',' << num(p.medianDurationSeconds) << ','
            << num(p.meanImpressions) << ',' << num(p.satisfactionPerMinute) << ','
            << num(p.regretPerMinute) << ',' << num(p.meanSessionUtility) << ','
            << num(p.earlyFailureExitRate) << ',' << num(p.naturalCompletionRate) << ','
            << num(p.harmfulFatigueMean) << ',' << num(p.nextSessionStartingSatisfaction) << ','
            << p.exits.failure << ',' << p.exits.satisfied << ',' << p.exits.fatigue << ','
            << p.exits.external << ',' << p.exits.regret << '\n';
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
    // Phase 15 welfare files (V2 TDD §6): written ONLY under realism.latent_reactions, so a
    // gate-off run's output directory carries no welfare CSVs and every EXISTING file is
    // byte-identical (D17).
    if (result.welfare.configured) {
        writeWelfareMetricsCsv(result);
        writeWelfareArchetypeMetricsCsv(result);
    }
    // Phase 16 session-health file (V2 TDD §4.9/§6): written ONLY under realism.session_dynamics,
    // so a P15-only or gate-off run's output directory carries no session_health.csv and every
    // EXISTING file (incl. the P15 welfare CSVs) is byte-identical (D17).
    if (result.sessionHealth.configured) {
        writeSessionHealthMetricsCsv(result);
    }
    writeMetadataJson(result.directory, meta);
}

} // namespace rr
