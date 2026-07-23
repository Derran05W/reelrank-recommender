# Phase 22 — Frozen Contracts (Training-Data Logging + Offline Learned Models)

Binding integration contract (P20/P21 pattern). Plan: `plan/06-PHASES-LEARNED-RANKING.md` Phase
22. TDD: V2 §4.19, §4.20 items 1–2, §4.22, §7 purity, §5. D21 is the binding model-scope
contract; D17–D25 apply.

## 1. Config surface (scaffold owns config.{hpp,cpp})

`learning_v2` block, all defaults preserving current behaviour:
- `training_log` : bool, default false. Load-validation: requires
  `simulation.scheduler=="event_queue"` (P22 logs the event-driven world only, V2 §9).
- `log_sample_rate` : double, default 0.25 — fraction of REQUESTS whose SHOWN impressions are
  logged (features + outcomes join).
- `log_pool_sample_rate` : double, default 0.01 — fraction of requests whose FULL RANKED POOL is
  logged (the §4.22 eligibility/position-bias support; ~500 rows per sampled request).
- `log_max_rows_per_file` : uint, default 2_000_000 — rotation threshold for candidates/outcomes
  part files (`-partNNNN.csv`).
- `survey.enabled` : bool, default false; `survey.sample_rate` : double, default 0.02;
  `survey.noise_sd` : double, default 0.35 (Likert-quantized noisy immediateSatisfaction).
- SAMPLING DRAWS NO RNG: request selection is the pinned SplitMix64-finalizer
  `hash01(requestId.value ^ kLogSampleSalt)` < rate (two distinct documented salts for the two
  rates) — zero stream perturbation, golden-tripwired like cohortHash01. The survey DOES draw
  (rate bernoulli + gaussian noise) on the D19-pinned `"explicit-feedback"` stream, exactly two
  draws per surveyed impression, zero when disabled.

## 2. Log layout (frozen; written under `<run-dir>/training_log/`)

- `schema.json` — {"schema_version": 1, feature column names in order, stage semantics, salts,
  config echo}. Written once per run.
- `requests.csv` — `request_id,user_id,session_id,timestamp,feed_size,effective_epsilon,`
  `pool_size,shown_count,pool_logged` (one row per SAMPLED request, both rates union).
- `candidates.csv` (+rotation parts) — one row per (request, pool candidate) for pool-sampled
  requests AND per (request, shown candidate) for shown-sampled requests:
  `request_id,reel_id,pool_rank,shown,position,served_score,exploration_flag,`
  `retrieval_sources,retrieval_similarity,` then the FEATURE COLUMNS: every field of
  `rr::FeatureVector` in declaration order, snake_case (`similarity,quality,freshness,trending,`
  `creator_affinity,duration_match,impression_penalty,repetition_penalty,exploration,`
  `session_topic,popularity,` + V2: `visual_match,music_match,emotional_match,`
  `emotional_intensity,production_quality,clickbait,information_density,language_match,`
  `usefulness,save_popularity`). `position` = feed slot for shown rows, −1 for pool-only rows.
  FEATURES ARE SERVED-TIME VALUES captured at the ranking call — never recomputed post-hoc.
- `outcomes.csv` (+parts) — SEPARATE label table (purity: evaluation-side observables only):
  `request_id,reel_id,position,watch_seconds,watch_ratio,completed,liked,shared,followed,`
  `not_interested,commented,saved,profile_visited,observed_exit_after_impression` — joined from
  InteractionEvents for SHOWN sampled impressions.
- `survey.csv` — gate `survey.enabled` only: `user_id,reel_id,request_id,timestamp,likert`
  (likert ∈ 1..5, quantized from immediateSatisfaction + gaussian(noise_sd), mapping documented).
  THE ONLY hidden-derived table, clearly labeled in schema.json.

## 3. Module placement (purity by construction, D18)

- The training LOGGER lives in `include/rr/learning_v2/training_logger.hpp` +
  `src/learning_v2/training_logger.cpp` — a NON-carve-out TU: the D18 include-graph guard must
  bar it from `simulation/hidden/` (scaffold extends the guard's scanned roots to
  `learning_v2`). It sees only (request, User, pool Candidates, FeatureVectors, RankedReels,
  InteractionEvents) — all recommender-visible/observable.
- The SURVEY writer lives in `src/evaluation/survey_writer.cpp` (carve-out, oracle-flavoured per
  D18; reads latent immediateSatisfaction).
- Hook sites (scaffold, event runner only): after ranking (features+pool+shown capture), after
  each impression's InteractionEvent (outcome append), run end (flush + schema.json).

## 4. Purity audit (package A; V2 §7 mandated)

`tests/integration/training_log_purity_test.cpp`: runs a TINY gate-on event sim with
training_log+survey on, then audits the EMITTED FILES: (a) every CSV header column ∈ the frozen
ALLOWLIST from `training_log_schema.hpp` (exact string sets — any new/renamed column fails);
(b) forbidden-substring scan on headers of feature/outcome tables (`latent`, `satisfaction`,
`regret`, `archetype`, `trust`, `hidden`, `fatigue`, `plasticity`, `tolerance`) — survey.csv
exempted for its documented columns only; (c) schema.json version + column echo matches.

## 5. Learners + offline eval (package B; D21 binding)

- `src/learning_v2/{logistic_regression,linear_regression}.{hpp,cpp}`: deterministic mini-batch
  SGD (fixed iteration order; shuffle via `"training-split"`-forked rr::Rng; init via
  `"model-init"`), z-score feature standardization fit on the train split and stored in the
  model; JSON serialization round-trip (D6).
- `src/learning_v2/training_data.{hpp,cpp}`: CSV reader for §2 files (join
  candidates×outcomes on (request_id,reel_id); shown rows only for training), split assignment:
  `temporal` (by request timestamp: first 80% train) and `user_disjoint` (pinned
  hash01(userId ^ kSplitSalt) < 0.8) — both deterministic, selected by CLI.
- `apps/train_models.cpp`: `--log-dir --out-dir --split temporal|user_disjoint --seed N
  [--targets csv-list] [--survey]` — trains all eight §4.19 targets (binary: completed, liked,
  shared, followed, not_interested, session_exit(=observed_exit_after_impression); linear:
  watch_ratio; survey-satisfaction regressor only with --survey and survey.csv present), writes
  per-target `model-<target>.json`, and appends `training_eval.csv`:
  `target,model,split,n_train,n_test,auc,log_loss,rmse,calibration_slope,calibration_intercept,`
  `base_rate` (NaN where inapplicable) for the learned model AND the three baselines: global
  frequency, per-source frequency (majority retrieval source label), served-score-as-predictor
  (`served_score` column). Also writes `calibration-<target>.csv` (10 equal-count bins:
  `bin,mean_pred,mean_actual,count`).
- AUC = deterministic rank-based (ties averaged); log-loss clamped p∈[1e-6,1−1e-6].

## 6. Package C — run script + report tooling (non-C++, fixtures only pre-integration)

- `scripts/run_phase22_logworld.sh`: ONE log-world run — `configs/realism-medium-retention.json`
  + `learning_v2.training_log=true` + survey enabled + `evaluation.ecosystem_metrics=false`,
  seed 42 (the pinned identical-world log per Tier-5 acceptance; document expected log sizes
  from §2 rates), then the train_models invocation for both splits.
- `scripts/phase22_report.py`: renders `results/published/phase22/offline_eval.md` from
  training_eval.csv (per-target table: learned vs three baselines, both splits; honest
  per-target verdicts incl. no-signal targets) + copies calibration CSVs.
- `scripts/plot_results.py` append-only: `plot_calibration(target_csvs, outdir)` (reliability
  diagrams, one PNG per target) and `plot_offline_auc(training_eval, outdir)` (grouped bar:
  learned vs baselines per target). Self-test with synthetic fixtures.

## 7. Tests (split)

- A: purity audit (§4); logging determinism (same seed twice ⇒ byte-identical training_log
  files); rotation correctness; survey draw-count/stream discipline (off ⇒ zero draws on
  "explicit-feedback"; on ⇒ exactly 2 per surveyed impression, V1 streams untouched);
  gate-off byte-identity (training_log=false ⇒ event-digest golden PASS + no training_log dir).
- B: learner convergence on synthetic separable data (unit, hand-checkable); split determinism
  (same seed ⇒ identical splits/batches); serialization round-trip bit-exact reload; offline
  learned-beats-frequency statistical test on a reduced-scale generated log (per-target where
  signal exists — SKIP-with-reason for targets whose base rate at reduced scale is <20
  positives, honest reporting; margins at demonstrated operating points, mechanism-vs-baseline
  only — NEVER fine ordering between two learned variants, P21 lesson).
- Golden obligations at integration: D17 small + drift-medium + event-digest (product code
  touched: runner hooks + orchestrator feature capture).

## 8. Ownership

- Scaffold: config block + validation; `include/rr/learning_v2/training_log_schema.hpp` (frozen
  column-name constants used by writer AND audit); logger/survey-writer stubs + event-runner
  hook sites (gate-guarded, no-op default); D18 guard extension to learning_v2; CMake
  (rr_learning_v2 lib + train_models app targets); this file.
- A (worktree): `src/learning_v2/training_logger.cpp` (+hpp internals),
  `src/evaluation/survey_writer.cpp` (+hpp), the runner hook bodies (event_driven_runner.cpp is
  A's this phase), A's tests.
- B (worktree): `src/learning_v2/{logistic_regression,linear_regression,training_data}.*`,
  `apps/train_models.cpp`, B's tests. Touches NOTHING A owns.
- C (main checkout): `scripts/run_phase22_logworld.sh`, `scripts/phase22_report.py`,
  plot_results.py append-only, report/publication templates.
- Contended-by-declaration: config files, CMakeLists (scaffold only), everything unlisted.
