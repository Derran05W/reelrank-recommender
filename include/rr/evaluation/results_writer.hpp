#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

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
    // Phase 19 (V2 TDD §4.13, D22). Written by writeAll ONLY under the event scheduler
    // (result.eventMode.configured); exposed here for targeted tests. Deterministic (fixed
    // precision, classic locale). serving_metrics.csv: per simulated day — day, feed_requests,
    // ranking_computations, impressions, stale_impressions, stale_impression_rate, mean_staleness,
    // satisfaction_lost. A round-robin run writes no serving_metrics.csv, so its output directory
    // is byte-identical to a pre-Phase-19 run (D17).
    static void writeServingMetricsCsv(const ExperimentResult &result);
    // Phase 20 (V2 TDD §4.15-4.17/§6, D22). Written by writeAll ONLY when a P20 gate is on
    // (result.longTerm.configured); exposed here for targeted tests. Deterministic (fixed
    // precision, classic locale). longterm_metrics.csv: per simulated day — day, sessions,
    // active_users, sessions_per_active_user, mean_session_satisfaction, mean_trust,
    // cumulative_churned, mean_pref_shift_from_initial. A gates-off run writes no
    // longterm_metrics.csv, so its output directory is byte-identical to a pre-Phase-20 run (D17).
    static void writeLongTermMetricsCsv(const ExperimentResult &result);
    // Phase 21 (contracts §2, D22). Written by writeAll ONLY when evaluation.ecosystem_metrics is
    // on (result.ecosystem.configured, event mode); exposed here for targeted tests. Deterministic
    // (fixed precision, classic locale). ecosystem_metrics.csv: per simulated day — the FROZEN
    // header day,impressions,creator_hhi,tail_creator_share,arch_genuinely_satisfying,arch_useful,
    // arch_ragebait,arch_clickbait,arch_comfort,arch_polished_irrelevant,arch_niche_treasure,
    // arch_background_music,niche_in_cohort_match_rate. A gate-off run writes no
    // ecosystem_metrics.csv, so its output directory is byte-identical to a run without the gate
    // (D17).
    static void writeEcosystemMetricsCsv(const ExperimentResult &result);
    // Phase 23 (contracts §3/§4, D22). Written by writeAll ONLY when learning_v2.learned_ranker is
    // on (result.learnedModels.configured, event mode); exposed here for targeted tests.
    // retraining_log.csv: the FROZEN header version,sim_time_seconds,n_train_rows,wall_ms,
    // targets_trained — one row per retrain (wall_ms is the ONLY non-deterministic column, D9).
    // A gate-off run writes no retraining_log.csv, so its output directory is byte-identical (D17).
    static void writeRetrainingLogCsv(const ExperimentResult &result);
    // Phase 23 (contracts §4/§6). A handful of served candidates' explanation maps from the first
    // learned-served feed (reel ids + feed positions + the §2 predicted_* / learned_value /
    // fallback / satisfaction_available terms), for package C to render — the self-describing
    // schema is written inline in the file. Gate-on only; deterministic.
    static void writeExplanationSampleJson(const ExperimentResult &result);

    // Per-user hidden-preference export row (Phase 20, contract §5). One row per user; the writer
    // emits them in the caller-provided order (the event runner sorts ascending user_id).
    struct HiddenPreferenceFinalRow {
        uint32_t userId = 0;
        double plasticity = 0.0;     // hidden preferencePlasticity trait
        bool churned = false;        // hidden retention.churned at run end
        double semanticShift = 0.0;  // 1 - cos(semantic p(0), p(T))
        double visualShift = 0.0;    // 1 - cos(visual p(0), p(T))
        double musicShift = 0.0;     // 1 - cos(music p(0), p(T))
        double emotionalShift = 0.0; // 1 - cos(emotional p(0), p(T))
        std::vector<double>
            semanticFinal; // final semantic preference components (sem_v0..sem_v{D-1})
    };
    // Phase 20 (contract §5, evaluation carve-out — D18-legal): the per-user counterfactual-
    // distortion export, written ONLY under a P20 gate (the caller guards on longTerm.configured).
    // Frozen header `user_id,plasticity,churned,sem_shift,visual_shift,music_shift,emotional_shift,
    // sem_v0..sem_v{D-1}`. Deterministic (fixed precision, classic locale). Standalone (not part of
    // writeAll) because it needs per-user hidden state the ExperimentResult does not carry, so the
    // runner that owns the hidden states supplies the rows.
    static void writeHiddenPreferenceFinalCsv(const std::filesystem::path &directory,
                                              const std::vector<HiddenPreferenceFinalRow> &rows);
};

} // namespace rr
