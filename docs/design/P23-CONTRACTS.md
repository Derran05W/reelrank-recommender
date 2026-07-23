# Phase 23 — Frozen Contracts (Learned Multi-Objective Ranking in the Loop)

Binding integration contract (P20–P22 pattern). Plan: `plan/06-PHASES-LEARNED-RANKING.md` Phase
23. TDD: V2 §4.21, Tier 5 acceptance (closed-loop half), §6, §10 item 8. D17–D25 apply; D21
still binds model scope. ORCHESTRATION NOTE: this phase has ONE C++ writer (package A, main
checkout, no worktree — it lands the config surface itself, first); packages B/C are non-C++ in
the same checkout with disjoint files and read THIS file for every key name.

## 1. Config surface (package A lands it FIRST, exactly these names)

`learning_v2` block additions (all defaults preserve current behaviour):
- `learned_ranker` : bool, default false. Validation: requires `training_log` (the ranker trains
  on the in-run log) — hence transitively event-mode.
- `value_weights` : nested block — `watch` (0.30), `share` (0.15), `follow` (0.10),
  `satisfaction` (0.30), `exit` (0.10), `regret` (0.05). §4.21 value =
  watch·pWatch + share·pShare + follow·pFollow + satisfaction·pSatisfaction − exit·pExit −
  regret·pRegret.
- `retrain_every_hours` : double, default 24 (simulated hours between retrains).
- `min_training_rows` : uint, default 5000 — below this, LearnedRanker serves the hand-tuned
  `WeightedRanker` scores (cold-start fallback; the explanation output marks `fallback=1`).
- `retrain_epochs` : uint, default 50 (in-loop SGD epochs; offline app default 200 unchanged).
- New algorithm string: `"hnsw_learned_ranker"` — the hnsw_ranker pipeline with LearnedRanker in
  place of WeightedRanker (same sources; diversity off by default, same as hnsw_ranker). Factory
  rejects it unless `learned_ranker` gate is on.

## 2. §4.21 term mapping (recorded design decisions — package A documents at definitions)

- pWatch = the watch_ratio LINEAR model's prediction clamped [0,1.5]/1.5-normalized (document).
- pShare / pFollow / pExit = the shared/followed/observed_exit_after_impression logistic models.
- pSatisfaction = the SURVEY-trained satisfaction regressor mapped from Likert 1–5 to [0,1]
  ((pred−1)/4, clamped). When survey data is absent/below min rows, the term contributes 0 and
  the explanation marks `satisfaction_available=0` (the survey-off experiment arm reads this).
- **pRegret := the not_interested logistic model's prediction.** Rationale (frozen): true regret
  is hidden (D18); not-interested is its closest sanctioned observable correlate (P14's
  satisfaction/regret-driven not-interested mechanism), and P22 measured it learnable (AUC
  0.66–0.69). Recorded as the §4.21 "designed observable regret proxy".
- Rare/no-signal targets: `followed` (P22: no learnable signal at 1.5% base rate) still trains
  but its weight default stays small; the P22 finding is cited at the weight default.
- Explanation output (V1 §14.4 parity): `featureContributions`-style map on served candidates
  with keys `predicted_watch, predicted_share, predicted_follow, predicted_satisfaction,
  predicted_exit, predicted_regret, learned_value, fallback, satisfaction_available` (weighted
  terms; `learned_value` = the sum; all present on every learned-served candidate).

## 3. In-loop retraining (package A)

- Deterministic schedule: retrain when simulated time crosses each `retrain_every_hours`
  boundary since run start, evaluated in the event runner at the P23 hook site; trains on the
  logger's IN-MEMORY shown-row matrix accumulated so far (the logger keeps this additional
  in-memory copy only when `learned_ranker` is on; ~10 MB at medium scale — documented).
- Training uses P22's learners with `retrain_epochs`, master-seed-derived "training-split"/
  "model-init" streams salted per version (document exact derivation) — same seed ⇒ bit-identical
  model sequence (test-enforced).
- Wall cost measured per retrain (steady_clock, OUTSIDE simulated time — D9 compliant) and
  recorded.
- Frozen output: `retraining_log.csv` — header
  `version,sim_time_seconds,n_train_rows,wall_ms,targets_trained` (targets_trained =
  pipe-delimited list); summary block `learned_models` — keys `configured` (bool),
  `retrain_count`, `final_version`, `total_retrain_wall_ms`, `mean_n_train_rows`,
  `fallback_request_share` (share of requests served by the cold-start fallback), `note`.
  Both emitted only when `learned_ranker` on (D22 additive).

## 4. Closed-loop experiment matrix (package B — script + methodology; integrator executes)

Base: `configs/realism-medium-retention.json` + training_log=true + survey.enabled=true + seed
42, event mode (identical worlds across ALL arms). `scripts/run_phase23_experiment.sh` generates
per-arm configs (established JSON-patch heredoc), ≤3 concurrent, arms:
- `hand_tuned` — algorithm hnsw_ranker (the baseline; also THE control for byte-identity: with
  learned_ranker off this is exactly the P22 log world → the script may symlink/reuse if
  present, else rerun).
- `semantic` — hnsw_ranker + the P15 semantic policy (all learned/tuned weights zeroed except
  similarity; copy the resolved patch from results/published/phase15/semantic/config.json).
- `learned` — hnsw_learned_ranker, default value_weights, survey on.
- `learned_survey_off` — learned with survey.enabled=false (does explicit feedback rescue the
  satisfaction axis?).
- Frontier sweep (5): `w_sat_100` (satisfaction .60, watch 0), `w_sat_70` (.42/.18),
  `w_balanced` (= learned defaults, reuse the `learned` arm), `w_watch_70` (.18/.42),
  `w_watch_100` (0/.60) — only watch/satisfaction vary (renormalized against the default
  total), other weights at defaults; plus `w_watch_100_noexit` (w_watch_100 with exit=regret=0
  — the pure-engagement vector for exit criterion 2).
Total 9 distinct runs (learned == w_balanced). Per-arm ~15–40 min Release.
`scripts/phase23_gap_analysis.py`: per-target table — offline held-out AUC/RMSE (from each
learned arm's own final-version models re-evaluated on its held-out log tail; simplest honest
method: run apps/train_models --log-dir <arm log> per arm post-hoc with the same seed and read
training_eval.csv) vs closed-loop deltas (arm's engagement/welfare/long_term vs hand_tuned) —
+ a divergence discussion section template (feedback-loop effects). Self-test on fixtures.

## 5. Tests (package A; §7 conventions, margins mechanism-vs-control only)

- Statistical (reduced scale, in-process): learned arm's closed-loop reward/impression within a
  calibrated band of hand_tuned AND ≥ the semantic control on engagement (demonstrated
  operating point; document); the pure-engagement vector vs balanced vector separate
  directionally on satisfaction (frontier mechanism tripwire).
- Retraining determinism: same seed twice ⇒ identical retraining_log.csv + bit-identical final
  model JSON dumps.
- Serving purity parity: an integration test asserting the FeatureVector the LearnedRanker
  scores equals the logged candidates.csv feature row for the same (request, reel) — same
  extractor output end-to-end (V2 §10 item 8 evidence).
- Explanation well-formed: all §2 keys present; learned_value == Σ weighted terms (1e-6).
- Gate-off: all three goldens byte-identical (integration re-runs digest + small + drift-medium).
- Cold-start fallback: below min_training_rows the served scores equal WeightedRanker's exactly.

## 6. Package C — plots + report (non-C++, fixtures pre-integration)

- `scripts/plot_results.py` append-only: `plot_multiobjective_frontier(runs, outdir)` — scatter
  x=reward/impression (engagement), y=mean hidden satisfaction, point size or third panel =
  retention_7d, one labeled point per arm (the §10-item-8 headline figure);
  `plot_offline_closedloop_gap(gap_csv, outdir)` — per-target offline-vs-closed-loop bars.
- `scripts/phase23_comparison.py` (pattern: phase20/21 renderers): all-arms table across the
  four §6 groups + long_term + `learned_models` block (frozen §3 keys) + the frontier table +
  verdict lines; writes comparison.{md,csv} to results/published/phase23. `--self-test`.
- Report/publication templates; an explanation-example extraction snippet (pull one served
  candidate's explanation map from a learned arm's log/summary — coordinate with A's report on
  where explanations are surfaced; if only in-memory, A exposes a tiny
  `--dump-explanation-sample` on simulate or writes `explanation_sample.json` per run under the
  gate — A DECIDES and documents; C reads whatever A lands, discovered from A's committed code,
  NOT guessed).

## 7. Ownership (all in the MAIN checkout; disjoint)

- A (C++, lands first-listed items before the rest): config.{hpp,cpp} additions (§1);
  src/learning_v2/learned_ranker.{hpp,cpp} + retrainer.{hpp,cpp} (+ training_logger in-memory
  matrix addition); factory registration; event-runner retrain hook; results_writer
  retraining_log.csv + learned_models block; explanation_sample.json emission (§6 choice);
  A's test files; CMake only if globbing misses something (report if so).
- B: scripts/run_phase23_experiment.sh, scripts/phase23_gap_analysis.py.
- C: scripts/phase23_comparison.py, plot_results.py append-only additions.
- Everything else contended-by-declaration. B/C validate config keys against THIS file (A may
  not have landed them when B/C start); the integrator re-validates generated configs at
  execution time.
