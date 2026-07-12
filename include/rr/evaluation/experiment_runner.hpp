#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "rr/evaluation/metrics_collector.hpp"
#include "rr/evaluation/run_metadata.hpp"
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
