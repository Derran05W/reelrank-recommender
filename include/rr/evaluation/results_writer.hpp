#pragma once

#include <filesystem>

#include "rr/evaluation/experiment_runner.hpp"
#include "rr/evaluation/run_metadata.hpp"

namespace rr {

// Serializes an ExperimentResult to the §26 output layout in `result.directory`:
//   config.json, summary.json, retrieval_metrics.csv, recommendation_metrics.csv,
//   diversity_metrics.csv, learning_curve.csv, regret_curve.csv, latency_metrics.csv,
//   metadata.json.
//
// Phase 9 diversity addition: diversity_metrics.csv (TDD 18.4) and a `diversity` block in
// summary.json are written UNCONDITIONALLY for every run from Phase 9 on - the diversity of ANY
// algorithm's feeds is measurable and the phase comparison needs baseline numbers, so unlike the
// Phase 8 injection files these are NOT gated on a config flag. This does not perturb the
// regression contract: it only ADDS one file + one summary key; every PRE-EXISTING file stays
// byte-identical to a Phase 8 run at the same seed.
//
// Phase 8 cold-start additions: when injection is configured (result.coldStart.configured), TWO
// extra deterministic files are written - new_user_curve.csv and new_reel_exposure.csv - and a
// `cold_start` block is added to summary.json. When injection is NOT configured, NONE of these
// appear and every file is byte-identical to a pre-Phase-8 run (the regression contract).
//
// Phase 15 welfare additions (V2 TDD §6, D22): when realism.latent_reactions is on
// (result.welfare.configured), TWO extra deterministic files are written - welfare_metrics.csv and
// welfare_archetype_metrics.csv - plus a `metric_groups` block and extra keys in the `welfare`
// block of summary.json (the four V2 §6 metric groups as SEPARATE blocks; no aggregate score). When
// the gate is off NONE of these appear and every EXISTING V1 file/column is byte-identical (D17);
// even gate-on, the existing V1 files/columns (recommendation_metrics.csv, diversity_metrics.csv,
// learning_curve.csv, the V1 summary blocks) are UNCHANGED — the V2 signals live only in the new
// files/blocks.
//
// Determinism (D8/TDD 24.6): every file EXCEPT latency_metrics.csv, metadata.json, and the
// `timing` subsection of summary.json is byte-identical across two runs with the same seed. This
// includes retrieval_metrics.csv and the two Phase-8 files: all come from deterministic
// computations. All floating-point output uses fixed precision under the classic locale so
// formatting never depends on the ambient locale.
class ResultsWriter {
  public:
    // Writes every §26 file. `meta` supplies the wall-clock/hardware provenance for metadata.json.
    static void writeAll(const ExperimentResult &result, const RunMetadata &meta);

    // Individual writers (exposed for targeted tests).
    static void writeConfigJson(const ExperimentResult &result);
    static void writeSummaryJson(const ExperimentResult &result);
    static void writeRetrievalMetricsCsv(const ExperimentResult &result);
    static void writeRecommendationMetricsCsv(const ExperimentResult &result);
    // Phase 9 (TDD 18.4). Written by writeAll UNCONDITIONALLY (diversity is measurable for any
    // algorithm); exposed here for targeted tests. Columns: round, mean_unique_topics,
    // mean_unique_creators, mean_intra_list_similarity, mean_topic_hhi, mean_creator_hhi,
    // repetition_rate. Deterministic (fixed precision, classic locale).
    static void writeDiversityMetricsCsv(const ExperimentResult &result);
    static void writeLearningCurveCsv(const ExperimentResult &result);
    static void writeRegretCurveCsv(const ExperimentResult &result);
    static void writeLatencyMetricsCsv(const ExperimentResult &result);
    // Phase 8 (TDD 18.5). Written by writeAll ONLY when result.coldStart.configured; exposed here
    // for targeted tests. new_user_curve.csv: impression_index, users_at_index, mean_reward,
    // mean_regret. new_reel_exposure.csv: round, injected_impressions, injected_impressions_cum,
    // distinct_injected_exposed_cum, share_of_round_impressions.
    static void writeNewUserCurveCsv(const ExperimentResult &result);
    static void writeNewReelExposureCsv(const ExperimentResult &result);
    // Phase 15 (V2 TDD §6, D22). Written by writeAll ONLY when result.welfare.configured (gate-on
    // under realism.latent_reactions); exposed here for targeted tests. Deterministic.
    //   welfare_metrics.csv: per round — round, impressions, mean_immediate_satisfaction,
    //   mean_regret, satisfaction_per_minute, watch_minutes, comment_rate, save_rate,
    //   profile_visit_rate, harmful_fatigue, platform_trust. (harmful_fatigue/platform_trust are
    //   NOT-YET-MODELED placeholders, constant 0 — real in P16/P20.)
    //   welfare_archetype_metrics.csv: per catalog archetype (index order) — archetype_index,
    //   archetype_name, impressions, exposure_share, mean_immediate_satisfaction, mean_regret.
    static void writeWelfareMetricsCsv(const ExperimentResult &result);
    static void writeWelfareArchetypeMetricsCsv(const ExperimentResult &result);
    // Phase 16 (V2 TDD §4.9/§6, D22). Written by writeAll ONLY when result.sessionHealth.configured
    // (gate-on under realism.session_dynamics); exposed here for targeted tests. Deterministic.
    //   session_health.csv: per round — round, sessions, open_sessions, mean_duration_seconds,
    //   median_duration_seconds, mean_impressions, satisfaction_per_minute, regret_per_minute,
    //   mean_session_utility, early_failure_exit_rate, natural_completion_rate,
    //   harmful_fatigue_mean, next_session_starting_satisfaction, then the exit-type counts
    //   failure_exits/satisfied_exits/ fatigue_exits/external_exits/regret_exits. All means/rates
    //   are over CLOSED sessions; RunEnded (open) sessions appear only in open_sessions. Under the
    //   P16 scaffold stub the loop closes zero sessions, so every row is a well-formed zero row
    //   (the populated path lands with package A's exit model).
    static void writeSessionHealthMetricsCsv(const ExperimentResult &result);
};

} // namespace rr
