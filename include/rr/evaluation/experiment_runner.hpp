#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "rr/evaluation/metrics_collector.hpp"
#include "rr/evaluation/run_metadata.hpp"
#include "rr/evaluation/session_health_metrics.hpp"
#include "rr/evaluation/welfare_metrics.hpp"
#include "rr/infrastructure/config.hpp"

namespace rr {

// Per-round row (one request per user per round). Bundles the §18.3 behaviour metrics for the
// round with the round's oracle-regret aggregates.
struct RoundMetrics {
    size_t round = 0;
    MetricsSummary metrics;
    size_t sampledRequests = 0; // requests whose Bernoulli(oracleSampleRate) draw fired this round
    double meanRegret = 0.0;    // mean regret over this round's sampled requests (0 if none)
    double cumulativeRegret = 0.0; // running sum of sampled-request regret through this round

    // Live retrieval metrics (TDD 18.1), means over this round's Bernoulli(retrievalSampleRate)
    // samples; all 0 for a 0-sample round or a non-vector algorithm (documented in output).
    size_t retrievalSamples = 0;
    double meanRecallAt10 = 0.0;
    double meanRecallAt50 = 0.0;
    double meanDistanceError = 0.0;

    // Estimate<->hidden alignment (TDD 18.5/18.6): mean over ALL users, measured at the END of this
    // round, of cos(estimatedPreference, hiddenPreference). Both vectors are unit-length, so this
    // is their dot product. This is an evaluation-side hidden-state read (TDD 18.2's carve-out, the
    // same as the per-impression trueAffinity): the aggregate never reaches a recommender. With
    // learning enabled it should trend up as estimates converge to the hidden preference; with
    // learning disabled (frozen arm) it is CONSTANT across rounds (estimates never change).
    double meanEstimatedHiddenCosine = 0.0;

    // Diversity metrics (Phase 9, TDD 18.4), means over THIS round's feeds - one feed per request,
    // measured on every request unsampled (the computation is trivial). `diversityFeeds` is the
    // number of feeds aggregated this round (== the round's request count). The five means come
    // from FeedDiversity (unique topics/creators, intra-list embedding cosine, topic/creator HHI).
    // `repetitionCount` is the total repeat items this round and `repetitionRate` is
    // repeats/total-feed-items - both expected 0 by construction (orchestrator seen-filter +
    // dedup), published as live verification of the "duplicate/repetitive content eliminated" exit
    // criterion. Deterministic (rng/clock-free): part of the byte-identical determinism guarantee.
    std::size_t diversityFeeds = 0;
    double meanUniqueTopics = 0.0;
    double meanUniqueCreators = 0.0;
    double meanIntraListSimilarity = 0.0;
    double meanTopicConcentration = 0.0;   // topic HHI
    double meanCreatorConcentration = 0.0; // creator HHI
    std::size_t repetitionCount = 0;
    double repetitionRate = 0.0;

    // Drift cohort split (Phase 10, TDD 18.6). Populated only when drift is configured; when it is
    // NOT, all remain zero and NONE of these are written (the regression contract). The split is by
    // DriftScheduler::everApplies(userId): "drifted" = users in at least one event's cohort,
    // "control" = the rest. Rewards are means over THIS round's impressions of the cohort;
    // alignments are means over the cohort's users of end-of-round cos(estimated, hidden) (the same
    // evaluation-only hidden-state read as meanEstimatedHiddenCosine, TDD 18.2 carve-out). The
    // *Impressions / *AlignUsers counts let the writer print `nan` for an empty cohort (e.g.
    // control is empty under whole-population drift) instead of a meaningless 0.0. Deterministic
    // (rng/clock-free): part of the byte-identical guarantee.
    std::size_t driftedImpressions = 0;
    double driftedMeanReward = 0.0;
    std::size_t controlImpressions = 0;
    double controlMeanReward = 0.0;
    std::size_t driftedAlignUsers = 0;
    double driftedAlignment = 0.0;
    std::size_t controlAlignUsers = 0;
    double controlAlignment = 0.0;
};

// One row of new_user_curve.csv (Phase 8, TDD 18.5): the mean reward and mean oracle regret over
// all INJECTED users at a given per-user impression index. `meanReward`/`meanRegret` are 0 when
// `usersAtIndex` is 0 (no injected user reached that index).
struct NewUserCurvePoint {
    std::size_t impressionIndex = 0;
    std::size_t usersAtIndex = 0;
    double meanReward = 0.0;
    double meanRegret = 0.0;
};

// One row of new_reel_exposure.csv (Phase 8, TDD 18.5 new-reel exposure): per-round impressions
// landing on injected reels, with cumulative totals and this round's share of all impressions.
struct NewReelExposurePoint {
    std::size_t round = 0;
    std::size_t injectedImpressions = 0;    // impressions on injected reels this round
    std::size_t injectedImpressionsCum = 0; // running total across rounds
    std::size_t distinctInjectedExposedCum =
        0;                                // distinct injected reels with >= 1 impression so far
    double shareOfRoundImpressions = 0.0; // injectedImpressions / all impressions this round
};

// Cold-start / injection report (Phase 8, TDD 18.5). Populated only when injection is configured
// (newUsers > 0 || newReels > 0); when `configured` is false the whole block is absent from output
// and the run is byte-identical to a pre-Phase-8 run (the regression contract).
struct ColdStartReport {
    bool configured = false;
    uint32_t newUsers = 0;
    uint32_t newUsersAt = 0;
    uint32_t newReels = 0;
    uint32_t newReelsAt = 0;

    // New-user regret over the first N impressions (TDD 18.5): the pooled mean oracle regret across
    // all injected users' impressions with index in [0, N). -1 sentinel when no injected-user
    // impressions fell in the window.
    double meanRegretFirst10 = -1.0;
    double meanRegretFirst25 = -1.0;
    double meanRegretFirst50 = -1.0;
    double meanRegretFirst100 = -1.0;

    // Interactions-to-reach-target-reward (TDD 18.5). The target is the run's overall PRE-INJECTION
    // mean reward per impression (impressions consumed in rounds < newUsersAt); `targetDefined` is
    // false when there were no pre-injection impressions (e.g. users injected at round 0).
    // `interactionsToTargetReward` is the smallest impression count K (1-based) at which the
    // injected users' cumulative mean reward over their first K impressions reaches the target; -1
    // when the target is undefined or never reached within the tracked window.
    bool targetDefined = false;
    double targetReward = 0.0;
    long interactionsToTargetReward = -1;

    // New-reel exposure totals over the whole run (TDD 18.5).
    std::size_t totalInjectedImpressions = 0;
    std::size_t distinctInjectedExposed = 0;
    double injectedImpressionShare = 0.0; // totalInjectedImpressions / all impressions in the run

    std::vector<NewUserCurvePoint> newUserCurve;
    std::vector<NewReelExposurePoint> newReelExposure;
};

// Preference-drift adaptation report (Phase 10, TDD 18.6). Populated only when drift is configured
// (config.drift has >= 1 event); when `configured` is false the whole block is absent from output
// and the run is byte-identical to a pre-Phase-10 run (the regression contract, mirroring
// ColdStartReport). Every field is deterministic (the scheduler is rng/clock-free, D8).
//
// The window anchor is `driftRound = firstDriftInteraction / feedSize` (integer floor): the first
// impression at 0-based index >= firstDriftInteraction lands in a feed served in round driftRound,
// so round driftRound is the FIRST round whose feeds can be affected by the drift, and rounds
// < driftRound are strictly pre-drift. All reward/alignment aggregates below are over the DRIFTED
// cohort only (the users whose hidden preference actually moved); the control cohort is the
// unaffected comparison group carried in the per-round RoundMetrics columns. Regret is measured in
// TRUE-AFFINITY units (Phase 4 oracle deviation; see the summary.json regret note).
//
// Sentinels: -1.0 for an undefined real-valued baseline (documented per field); -1 for a
// long "never happened / not applicable" round or interaction count.
struct AdaptationReport {
    bool configured = false;

    std::size_t driftedUsers = 0; // users with drift.everApplies(id) == true (final population)
    std::size_t controlUsers = 0; // the rest

    uint32_t firstDriftInteraction = 0; // scheduler's earliest configured atInteraction
    long driftRound = 0;                // firstDriftInteraction / feedSize (floor); anchor round
    std::size_t feedSize = 0;           // recorded so consumers can reconstruct the interaction map

    // Pre-drift reward baseline: mean of the drifted cohort's per-round mean reward over up to the
    // 3 rounds immediately before driftRound ([max(0, driftRound-3), driftRound-1]); fewer rounds
    // when drift is early. -1.0 when driftRound == 0 (no pre-drift rounds) or none of the window
    // rounds had drifted impressions -> then rewardDrop / recovery are undefined too.
    double preDriftReward = -1.0;

    // Trough = min drifted-cohort round mean reward over rounds >= driftRound (rounds with drifted
    // impressions only); troughRound is its round (-1 if there are no post-drift drifted rounds).
    // rewardDrop = preDriftReward - troughReward (0.0 when preDriftReward or troughRound
    // undefined).
    double troughReward = 0.0;
    long troughRound = -1;
    double rewardDrop = 0.0;

    // Recovery = first round >= driftRound whose drifted-cohort mean reward >= 0.95 *
    // preDriftReward; recoveryInteractions = (recoveryRound - driftRound + 1) * feedSize
    // (interactions the drifted cohort consumed from the drift round through recovery, inclusive).
    // Both -1 when preDriftReward is undefined (<= 0) or the threshold is never re-crossed within
    // the run.
    long recoveryRound = -1;
    long recoveryInteractions = -1;

    // Alignment (est<->hidden cosine) adaptation of the drifted cohort. preDriftAlignment is the
    // cohort's alignment at round driftRound-1 (-1.0 when driftRound == 0). postDriftAlignmentMin
    // is the min over rounds >= driftRound (its round in *Round; -1 if none).
    // alignmentRecoveryRound is the first round >= driftRound with alignment >= 0.95 *
    // preDriftAlignment -- the threshold crossing read as "interactions until the new preference is
    // detected" (the estimate has re-locked onto the drifted hidden preference); -1 when undefined
    // or never.
    double preDriftAlignment = -1.0;
    double postDriftAlignmentMin = 0.0;
    long postDriftAlignmentMinRound = -1;
    long alignmentRecoveryRound = -1;

    // Cumulative regret during adaptation: sum of the run's per-round SAMPLED oracle regret over
    // rounds [driftRound, recoveryRound] (recoveryRound replaced by the last round when recovery
    // never happens). This is the whole-population sampled aggregate (regret is not cohort-split in
    // the harness), in true-affinity units. 0.0 when driftRound is beyond the run.
    double adaptationWindowRegret = 0.0;
};

// Hidden-user-welfare metrics (WelfareReport / WelfareRoundPoint / ArchetypeWelfare) are defined in
// rr/evaluation/welfare_metrics.hpp (the V2 §6 welfare-group module, Phase 15 — extending the lean
// Phase 14 slice). They remain in scope here via that include, so consumers of this header are
// unaffected by the move.

// Phase 18 event-mode additions (D20/D22). `configured` is false for the round-robin scheduler, so
// every field stays inert and NO new summary key is written — round-robin output stays
// byte-identical (D17 golden is the tripwire). Under simulation.scheduler == "event_queue" the
// EventDrivenRunner fills it: the deterministic event-log digest + event count (package C's golden
// tripwire, folded by rr::foldEventLog over the full event stream) plus the event-mode
// session-health additions the round-robin loop cannot produce — sessions per simulated day, mean
// concurrent-online occupancy (fraction of users online, sampled once per processed event
// timestamp), and the baseline return-delay stats. Purely additive: ResultsWriter emits an
// `event_mode` summary block ONLY when configured, so the round-robin path is untouched.
// One row of serving_metrics.csv (Phase 19, D22 additive): the per-simulated-day serving/cost/
// staleness view package C plots. Written ONLY under the event scheduler; a round-robin run emits
// no serving_metrics.csv at all (byte-identity, D17). Deterministic (fixed precision, classic
// locale).
struct ServingDayPoint {
    std::size_t day = 0;
    std::size_t feedRequests = 0;
    std::uint64_t rankingComputations = 0;
    std::size_t impressions = 0;
    std::size_t staleImpressions = 0;
    double staleImpressionRate = 0.0; // staleImpressions / impressions (0 for an empty day)
    double meanStaleness = 0.0;       // mean over the day's impressions of applies-since-ranking
    double satisfactionLost = 0.0; // Σ clamped (day fresh-mean satisfaction − stale satisfaction)
};

struct EventModeReport {
    bool configured = false;
    uint64_t eventLogDigest = 0;
    std::size_t eventCount = 0;
    double simulatedDays = 0.0;           // config.simulation.horizonSeconds / 86400
    double sessionsPerSimulatedDay = 0.0; // closed sessions / simulatedDays
    double meanConcurrentOnline = 0.0;    // mean online fraction over sampled event timestamps
    double returnDelayMeanSeconds = 0.0;
    double returnDelayMedianSeconds = 0.0;
    std::size_t returnCount = 0; // number of baseline return-delay draws (scheduled returns)

    // --- Phase 19 serving / cost / staleness instrumentation (V2 §4.13, D22 additive) -----------
    // Filled by the EventDrivenRunner alongside the P18 fields above; ALL defaults are inert, so a
    // round-robin run (eventMode.configured == false) writes none of it and stays byte-identical
    // (D17). With DEFAULT serving (prefetch_depth 0, refill_threshold 0, invalidate off) the
    // numbers are still emitted, but the underlying event stream — and therefore the digest — is
    // unchanged. These back the freshness-versus-cost frontier (package C reads them per run + per
    // day).
    uint32_t servingPrefetchDepth = 0; // EFFECTIVE depth ranked per RequestFeed (0-config resolved
                                       // to recommendation.feed_size); the frontier's x-axis label
    uint32_t servingRefillThreshold = 0; // refill when remaining <= this (0 = refill-when-empty)
    bool servingInvalidateOnIntentChange = false;

    std::size_t feedRequestCount = 0; // feed requests served (== ExperimentResult::requestCount)
    std::uint64_t rankingComputations = 0; // Σ candidatesRanked over all feed requests (the COST)
    double meanStaleness = 0.0;       // mean over impressions of (updater applies since ranking)
    double staleImpressionRate = 0.0; // fraction of impressions with staleness > 0
    std::size_t staleImpressionCount = 0;
    // Cumulative satisfaction lost during stale serving windows: Σ over stale (staleness>0)
    // impressions of max(0, dayFreshMeanSatisfaction − thisImpressionSatisfaction), where
    // dayFreshMeanSatisfaction is the per-simulated-day mean immediate satisfaction of that day's
    // staleness-0 impressions (the fresh-serving reference). >= 0 by construction; documented
    // definition in the runner.
    double satisfactionLostBeforeRefresh = 0.0;
    std::size_t feedInvalidationCount = 0; // intent-swing invalidations (0 unless invalidate on)

    // Adaptation delay after drift (P10-style, on satisfaction; only when config.drift is
    // configured). `adaptationConfigured` mirrors DriftScheduler::configured(); when false the
    // remaining fields stay 0 and the writer omits the sub-block. Delays are in per-user
    // interactions, aggregated over the drifted cohort's users that recovered within the horizon.
    bool adaptationConfigured = false;
    std::size_t adaptationDriftedUsers = 0;   // drifted-cohort users with enough pre-drift history
    std::size_t adaptationRecoveredUsers = 0; // of those, the ones that recovered within the run
    double meanAdaptationDelayInteractions = 0.0;
    double medianAdaptationDelayInteractions = 0.0;

    // Per-simulated-day serving rows for serving_metrics.csv (package C's per-day plots). Empty for
    // a round-robin run (no file written).
    std::vector<ServingDayPoint> servingByDay;
};

// Phase 20 long-term metrics (V2 TDD §4.15–4.17/§6, D22 additive). FROZEN SCHEMA (contracts §5):
// package B fills these, package C consumes them, so the field names + comments are the shared
// cross-package surface — do not rename without an orchestrator sign-off. `configured` is false
// unless a P20 gate is on, in which case NO `long_term` summary block and NO longterm_metrics.csv
// are written and the run is byte-identical to a pre-Phase-20 run (D17). LongTermDayPoint is
// defined FIRST so LongTermReport::byDay is a vector of a complete type.
struct LongTermDayPoint {
    uint32_t day = 0; // simulated day index from run start
    uint64_t sessions = 0;
    uint64_t activeUsers = 0; // users with >=1 session that day
    double sessionsPerActiveUser = 0.0;
    double meanSessionSatisfaction = 0.0;
    double meanTrust = 0.0; // mean over ALL users at day end (uninitialized trust reads
                            // as the user's platformTrust trait)
    uint64_t cumulativeChurned = 0;
    double meanPreferenceShiftFromInitial = 0.0; // as-of day end
    // Phase 21 (contracts §3): per-day mean per-user softmax topic-similarity entropy, snapshotted
    // at day end — the SAME formula + temperature constant as the run-end
    // mean_final_preference_entropy (the shared preferenceEntropy helper in
    // event_driven_runner.cpp). The interest-diversity trajectory the failure-mode scenarios plot.
    // Emitted as the TRAILING mean_preference_entropy column of longterm_metrics.csv (appended last
    // so existing readers keep working).
    double meanPreferenceEntropy = 0.0;
};

struct LongTermReport {
    bool configured = false;          // true iff preference_evolution || retention.enabled
    bool retentionConfigured = false; // true iff retention.enabled (event mode)
    double retention1d = 0.0;         // fraction of users with >=1 session STARTING in
                                      // (userFirstDayEnd, +1 day]; userFirstDayEnd = end of the
                                      // simulated day containing the user's first session
    double retention7d = 0.0;         // same with +7 days
    double sessionsPerUserPerDay = 0.0;
    double satisfactionWeightedRetention = 0.0; // sum_u(retained7d_u * satbar_u)/sum_u(satbar_u),
                                                // satbar_u = max(0, user mean session satisfaction)
    double churnRate = 0.0;                     // churned users / users
    double meanChurnProbability = 0.0;          // mean of model churnProbability at run end
    double meanFinalTrust = 0.0;
    double meanFinalHabit = 0.0;
    double meanPreferenceShiftFromInitial = 0.0; // mean_u (1 - cos(p_u(0), p_u(T))), semantic
    double meanFinalPreferenceEntropy = 0.0;     // mean_u entropy of softmax over topic-centre
                                                 // cosine similarities (documented in impl)
    std::vector<LongTermDayPoint> byDay;         // longterm_metrics.csv rows
};

// Phase 21 ecosystem failure-mode metrics (contracts §2, V2 TDD §4.18/§6, D22 additive). FROZEN
// SCHEMA: the scaffold fills these under the evaluation.ecosystem_metrics gate; the scenario
// packages (non-C++) consume the frozen keys, so field names + comments are the shared
// cross-package surface. `configured` is false unless evaluation.ecosystem_metrics is on (which
// itself requires the event scheduler), in which case NO `ecosystem` summary block and NO
// ecosystem_metrics.csv are written and the run is byte-identical to a run with the gate off (D17).
// Everything is computed in the D18 evaluation carve-out (reads HiddenReelState.archetypeIndex +
// niche band + rr::cohortHash01 on the hidden user id). kEcosystemArchetypeCount == 8 mirrors the
// shipped catalog (V2 TDD 4.4), whose index order the frozen CSV columns pin; the scenarios keep
// that catalog.
inline constexpr std::size_t kEcosystemArchetypeCount = 8;

struct EcosystemDayPoint {
    uint32_t day = 0; // simulated day index from run start (contiguous, zero-impression days
                      // emit a zero row for day-index continuity)
    uint64_t impressions =
        0; // that day's total impressions (disambiguates the 0-valued rates below)
    double creatorHhi = 0.0; // Σ_c (that-day impressions of creator c / that-day total)²
    // That day's impression share of creators OUTSIDE the top decile by CUMULATIVE impressions as
    // of end-of-day (a small/new-creator proxy — no creator injection exists in event mode). See
    // the runner for the exposed-creator-set + floor(N*0.1) + id-tiebreak definition.
    double tailCreatorShare = 0.0;
    // That day's impression share by hidden archetype, catalog index order (frozen 8: genuinely
    // satisfying, useful, ragebait, clickbait, comfort, polished_irrelevant, niche_treasure,
    // background_music). 0 for a zero-impression day.
    std::array<double, kEcosystemArchetypeCount> archShare{};
    // Among that day's niche impressions (reels with an active hidden niche band, i.e. the
    // niche_treasure archetype), the share where rr::cohortHash01(userId) falls in the reel's
    // hidden [centre-width, centre+width] band; 0 when the day has no niche impressions.
    double nicheInCohortMatchRate = 0.0;
};

struct EcosystemReport {
    bool configured = false;
    double creatorHhiFinalDay = 0.0; // last simulated day's creator_hhi
    double creatorHhiWholeRun = 0.0; // creator HHI over the whole run's impressions
    double tailCreatorShareWholeRun =
        0.0; // tail share over the whole run (decile by whole-run cum)
    std::array<double, kEcosystemArchetypeCount> archShareWholeRun{}; // whole-run archetype shares
    double nicheInCohortMatchRateWholeRun = 0.0;                      // whole-run niche match rate
    std::vector<EcosystemDayPoint> byDay; // ecosystem_metrics.csv rows (one per simulated day)
};

// Phase 23 learned-ranking report (contracts §3/§4, D22 additive). FROZEN SCHEMA: package C
// consumes these keys. `configured` is false unless learning_v2.learned_ranker is on (event mode),
// in which case NO retraining_log.csv, NO explanation_sample.json, and NO `learned_models` summary
// block are written and the run is byte-identical to a gate-off run (D17). Filled by the event
// runner.

// One retraining_log.csv row (contracts §3, frozen header
// version,sim_time_seconds,n_train_rows,wall_ms,targets_trained). `wallMs` is steady_clock (D9 —
// outside simulated time), so it is the ONE non-deterministic field and is excluded from the §5
// determinism comparison (like latency_metrics.csv); every other column is bit-reproducible.
struct RetrainRecord {
    int version = 0;
    std::uint64_t simTimeSeconds = 0; // the triggering RequestFeed's simulated time
    std::size_t nTrainRows = 0;       // complete rows the version trained on
    double wallMs = 0.0;              // steady_clock retrain cost (NOT part of determinism)
    std::string targetsTrained;       // pipe-delimited trained targets (absent targets omitted)
};

// One served candidate's explanation for explanation_sample.json (contracts §4/§6). The map is the
// LearnedRanker's §2 featureContributions (predicted_* weighted terms + learned_value + fallback +
// satisfaction_available). Captured from the FIRST learned-served feed after the models go ready.
struct ExplanationSampleCandidate {
    std::uint32_t reelId = 0;
    std::size_t position = 0; // feed slot
    std::unordered_map<std::string, float> explanation;
};

struct LearnedModelsReport {
    bool configured = false;
    int retrainCount = 0;              // number of actual retrains (== final version)
    int finalVersion = 0;              // last served model version (0 if never retrained)
    double totalRetrainWallMs = 0.0;   // Σ wall_ms (timing; not deterministic)
    double meanNTrainRows = 0.0;       // mean n_train_rows over retrains
    double fallbackRequestShare = 0.0; // share of feed requests served by the cold-start fallback
    std::string note;
    std::vector<RetrainRecord> retrains; // retraining_log.csv rows (in version order)

    // Final model bundle JSON dump — the retraining-determinism test compares this bit-for-bit
    // across two same-seed runs (contracts §5). In-memory only (not written to disk).
    std::string finalModelJson;

    // explanation_sample.json payload (deterministic: the first learned-served feed).
    bool explanationCaptured = false;
    std::uint64_t explanationRequestId = 0;
    std::uint32_t explanationUserId = 0;
    std::uint64_t explanationSimTimeSeconds = 0;
    int explanationVersion = 0;
    std::vector<ExplanationSampleCandidate> explanationCandidates;
};

// Everything one experiment produced, in memory. The ResultsWriter serializes it to disk; the
// simulate CLI prints headline lines from it. `directory` is the created <experiment-id> dir.
struct ExperimentResult {
    ExperimentConfig config;
    uint64_t seed = 0;
    std::string experimentId;
    std::filesystem::path directory;

    size_t userCount = 0;
    size_t reelCount = 0;
    size_t requestCount = 0;
    size_t impressionCount = 0;

    MetricsSummary overall;

    double oracleSampleRate = 0.0;
    size_t sampledRequestCount = 0;
    double meanRegret = 0.0;
    double cumulativeRegret = 0.0;

    // Online learning (Phase 7). `learningEnabled` mirrors config.learning.enabled; when false the
    // three per-user preference vectors are frozen at the cold-start prior (pre-Phase-7 behaviour).
    // `finalEstimatedHiddenCosine` is the estimate<->hidden alignment at the END of the last round
    // (rounds.back().meanEstimatedHiddenCosine; 0 if there are no rounds) -- the headline "did new
    // users converge toward their hidden preference" number (TDD 18.5).
    bool learningEnabled = true;
    double finalEstimatedHiddenCosine = 0.0;

    // Live retrieval metrics (TDD 18.1). `retrievalApplicable` is true iff the recommender is
    // vector-based (retrievalIndex() != nullptr); when false, no samples are taken and a note is
    // written. Overall recall/distance-error are means over all sampled requests (0 if none).
    bool retrievalApplicable = false;
    double retrievalSampleRate = 0.0;
    size_t retrievalSampleCount = 0;
    double retrievalRecallAt10 = 0.0;
    double retrievalRecallAt50 = 0.0;
    double retrievalDistanceError = 0.0;

    // Overall diversity (Phase 9, TDD 18.4): means over ALL feeds in the run + the run's total
    // repeat count. `diversityFeedCount` is the number of feeds measured (== requestCount).
    // `totalRepetitions` is expected 0 by construction; `repetitionRate` is the run's pooled
    // repeats/total-feed-items. Deterministic; surfaced in diversity_metrics.csv (per round) and
    // the summary.json `diversity` block (these overall means). Emitted for EVERY run from Phase 9
    // on - unlike the Phase 8 injection files, diversity_metrics.csv is UNCONDITIONAL (the
    // diversity of any algorithm's feeds is the phase-comparison baseline).
    std::size_t diversityFeedCount = 0;
    double meanUniqueTopics = 0.0;
    double meanUniqueCreators = 0.0;
    double meanIntraListSimilarity = 0.0;
    double meanTopicConcentration = 0.0;   // topic HHI
    double meanCreatorConcentration = 0.0; // creator HHI
    std::size_t totalRepetitions = 0;
    double repetitionRate = 0.0;

    std::vector<RoundMetrics> rounds;

    // Wall-clock, confined to summary.timing + latency_metrics.csv (D9/D8 determinism carve-out).
    // `latency` is the whole recommend() call; the three stage stats decompose it (TDD 18.7 /
    // Phase 5 exit criterion). Stage stats are all-zero for recommenders that do not populate the
    // per-stage response fields (Random/Popularity).
    LatencyStats latency;
    LatencyStats retrievalLatency;
    LatencyStats rankingLatency;
    LatencyStats rerankingLatency;
    double totalWallSeconds = 0.0;

    // Cold-start / injection metrics (Phase 8, TDD 18.5). `configured` is false for a normal run,
    // in which case no injection files/keys are written (byte-identical to a pre-Phase-8 run).
    ColdStartReport coldStart;

    // Preference-drift adaptation metrics (Phase 10, TDD 18.6). `configured` is false for a run
    // with no drift events, in which case no adaptation keys/columns are written (byte-identical to
    // a pre-Phase-10 run).
    AdaptationReport adaptation;

    // Hidden-user-welfare metrics group (Phase 15, V2 TDD §6, D22). `configured` is false for a
    // gate-off run (realism.latent_reactions off), in which case no welfare block/CSVs are written
    // (byte-identical to a pre-Phase-14 run, D17). Under the gate it carries the per-round +
    // overall satisfaction/regret/satisfaction-per-minute plus the per-archetype exposure
    // breakdown; the engagement group's V2 additions (comment/save/profile rates) live in
    // `overall`/`rounds`.
    WelfareReport welfare;

    // Session-health metric group (Phase 16, V2 TDD §4.9/§6, D22). `configured` is false unless
    // realism.session_dynamics is on, in which case no session_health block/CSV is written and the
    // run is byte-identical to a pre-Phase-16 run (D17). Under the gate it carries the per-round +
    // overall session statistics reduced from the exit-aware loop's collected SessionRecords: exit
    // counts/shares, time-before-exit (mean/median), satisfaction-/regret-per-minute, mean session
    // utility U_s, early-failure-exit and natural-completion rates, next-session starting
    // satisfaction, and the harmful-fatigue mean (which also realizes the welfare group's
    // previously-placeholder harmful_fatigue column). The P16 statistical tests and package C's
    // comparison read this in-process.
    SessionHealthReport sessionHealth;

    // Phase 18 event-mode additions (D20/D22). `configured` is false for the round-robin scheduler
    // (inert, no summary key written — byte-identical, D17); under the event scheduler it carries
    // the event-log digest + count and the event-mode session-health additions. See
    // EventModeReport.
    EventModeReport eventMode;

    // Phase 20 long-term metrics group (V2 TDD §4.15–4.17/§6, D22). `configured` is false unless a
    // P20 gate (realism.preference_evolution || retention.enabled) is on, in which case no
    // `long_term` block / longterm_metrics.csv is written and the run is byte-identical to a
    // pre-Phase-20 run (D17). Package B fills it under the gate; package C consumes the frozen §5
    // schema. See LongTermReport.
    LongTermReport longTerm;

    // Phase 21 ecosystem failure-mode metrics (contracts §2, D22 additive). `configured` is false
    // unless evaluation.ecosystem_metrics is on (event mode only), in which case no `ecosystem`
    // block / ecosystem_metrics.csv is written and the run is byte-identical to a gate-off run
    // (D17). The event runner fills it from the D18 evaluation carve-out; the scenario packages
    // consume the frozen keys. See EcosystemReport.
    EcosystemReport ecosystem;

    // Phase 23 learned-ranking report (contracts §3/§4, D22 additive). `configured` is false unless
    // learning_v2.learned_ranker is on (event mode), in which case no retraining_log.csv /
    // explanation_sample.json / `learned_models` block is written and the run is byte-identical to
    // a gate-off run (D17). Filled by the event runner from the LearnedRanker + Retrainer. See
    // LearnedModelsReport.
    LearnedModelsReport learnedModels;
};

// Runs the end-to-end evaluation loop (TDD 20 + phase-4 task 4, phase-7 tasks 1/4) from a
// fully-resolved config and writes the §26 output layout under <outputRoot>/<experiment-id>/.
//
// Flow: generateDataset -> cold-start prior -> interleaved requestsPerUser rounds over all users ->
// per-impression behaviour metrics + §18.2 true affinity -> ONLINE preference update after each
// interaction (OnlineUserStateUpdater, when config.learning.enabled) -> per-round estimate<->hidden
// alignment (TDD 18.5) -> oracle regret on a Bernoulli-sampled subset -> results on disk. With
// learning disabled the three per-user preference vectors stay frozen at the cold-start prior.
//
// The master seed is config.simulation.seed; independent named rng streams (behaviour /
// recommender / oracle, D8) keep the oracle from perturbing the simulation.
class ExperimentRunner {
  public:
    ExperimentRunner(ExperimentConfig config, std::filesystem::path outputRoot,
                     BuildProvenance provenance = {});

    // Generate, simulate, aggregate, and write all output files. Returns the in-memory result.
    ExperimentResult run();

  private:
    ExperimentConfig config_;
    std::filesystem::path outputRoot_;
    BuildProvenance provenance_;
};

} // namespace rr
