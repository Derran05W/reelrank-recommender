#!/usr/bin/env bash
#
# run_phase22_logworld.sh — Phase 22 Tier-5 log-world run (V2 TDD §4.19-4.20/§4.22, plan Phase 22
# task 5, docs/design/P22-CONTRACTS.md §6 [this package's own spec] + §2/§5).
#
# ONE arm, ONE pinned "log world": configs/realism-medium-retention.json (the SAME 10k-user/
# 100k-reel/32-topic/64-dim event-mode dataset, evolution+retention gates on, 9-simulated-day
# horizon, serving.prefetch_depth=10 world every Phase 20/21 experiment already shares) plus a
# small `learning_v2` patch that turns the training-log pipeline and survey on:
#
#   learning_v2.training_log            = true
#   learning_v2.log_sample_rate         = 0.25   (contracts §1 default -- spelled out explicitly)
#   learning_v2.log_pool_sample_rate    = 0.01   (contracts §1 default -- spelled out explicitly)
#   learning_v2.survey.enabled          = true
#   evaluation.ecosystem_metrics        = false  (contracts §6 literal text; ALREADY the base
#                                                  config's value since it omits the key entirely
#                                                  -- config.cpp defaults EvaluationConfig.
#                                                  ecosystemMetrics to false -- so this is a
#                                                  zero-behavior-change, explicit-for-documentation
#                                                  write, not a drift)
#
# Every OTHER `learning_v2` key (log_max_rows_per_file, survey.sample_rate, survey.noise_sd) and
# every OTHER top-level block (serving/evolution/retention/ranking/exploration/diversity/
# realism/...) is left COMPLETELY UNTOUCHED -- this script's python3 JSON-patch step below only
# ever ADDS the five keys listed above (never a wholesale block replacement, same discipline
# scripts/run_phase20_experiment.sh's ranking-weight patches document and this package inherits),
# so every other field resolves through config.cpp's defaults exactly as the committed base file
# already does. Algorithm is `hnsw_ranker` (contracts §6), passed on the `simulate` CLI (like every
# prior phase's scripts) rather than baked into the config file.
#
# CONFIG-KEY CROSS-CHECK (this package does not touch config.cpp/config.hpp -- verified by reading
# them directly, HEAD 67fe401): every key this script's patch writes is present in
# src/infrastructure/config.cpp's ensureKnownKeys() allowlists --
#   learning_v2         : {training_log, log_sample_rate, log_pool_sample_rate,
#                           log_max_rows_per_file, survey}                       (config.cpp:644-646)
#   learning_v2.survey  : {enabled, sample_rate, noise_sd}                      (config.cpp:629)
#   evaluation          : {oracle_sample_rate, retrieval_sample_rate, ecosystem_metrics}
#                                                                                (config.cpp:358-359)
# `learning_v2.training_log`/`survey.enabled` load-validation (config.cpp:742-749) requires
# `simulation.scheduler=="event_queue"`, already true in the base config -- no other gate
# dependency applies (unlike e.g. retention/preference_evolution, learning_v2 has no realism.*
# prerequisite).
#
# EXPECTED LOG SIZES (arithmetic from contracts §1-2's rates + REAL measured counts for this EXACT
# base config -- this worktree cannot run a real Phase 22 log-world experiment itself: package A's
# TrainingLogger::onRequestRanked/onImpressionOutcome are no-op stubs here, see
# src/learning_v2/training_logger.cpp, so there is no Phase 22 file to measure directly). Phase 21
# ran this SAME base config to completion for real, multiple times, at the SAME scale/algorithm/
# seed (only `evaluation.ecosystem_metrics` differs, which does not affect impression volume):
# results/phase21/niche_starvation/control/hnsw_ranker-seed42-20260719T090016/summary.json and
# results/phase21/popularity_feedback/control/hnsw_ranker-seed42-20260719T085051/summary.json both
# report counts.impressions=440616, counts.requests=112407. Using those REAL counts (tighter
# grounding than a round "~2M impressions" planning guess):
#
#   requests.csv rows -- union of the two INDEPENDENT pinned-hash draws (contracts §1: distinct
#   salts, so shown-sample and pool-sample selection are independent per request):
#     P(shown-sampled OR pool-sampled) = 1 - (1 - 0.25)(1 - 0.01) = 1 - 0.7425 = 0.2575
#     0.2575 x 112,407 requests ~= 28,945 rows
#
#   candidates.csv rows (contracts §2 header: 9 prefix cols + 21 feature cols = 30 cols/row):
#     shown-sample contribution : 0.25 x 440,616 impressions           ~= 110,154 rows
#     pool-sample contribution  : 0.01 x 112,407 requests x ~500/row   ~= 562,035 rows
#                                                     simple additive total ~= 672,189 rows
#     (the true count is a shade lower: a request that wins BOTH draws has its shown candidates
#     already inside its own pool dump, so they are not double-added; that overlap is
#     ~0.25 x 0.01 x 112,407 x 3.92 mean-shown/request ~= 1,100 rows, <0.2% of the total --
#     immaterial for a sizing estimate). Both well under log_max_rows_per_file's default
#     2,000,000 -- NO `-partNNNN` rotation is expected at this run's scale (single
#     candidates.csv/outcomes.csv files; the rotation path itself is package A's unit-test
#     territory, not exercised by this particular run's volume).
#
#   outcomes.csv rows -- gated by the SHOWN-sample draw only (training_logger.hpp's
#   onImpressionOutcome docstring: "Appends the observable label row to outcomes.csv for SHOWN
#   sampled impressions" -- the pool-sample draw does not gate outcomes):
#     0.25 x 440,616 ~= 110,154 rows
#
#   survey.csv rows -- this run sets survey.enabled=true at its 0.02 sample_rate default.
#   config.hpp documents the rate as a fraction of "shown impressions" (the WHOLE-RUN population,
#   an independent real-RNG draw on the pinned "explicit-feedback" stream per contracts §1 -- NOT
#   gated by the log's own pinned-hash shown-sample selection):
#     0.02 x 440,616 shown impressions ~= 8,812 total survey rows
#   of which only the subset whose request ALSO independently wins the log's shown-sample draw
#   joins onto candidates.csv by (request_id, reel_id) for the survey-satisfaction training
#   target: 0.02 x 0.25 x 440,616 ~= 2,203 joinable rows -- a modest but workable sample for one
#   linear-regression target; read that target's row in training_eval.csv with its own
#   n_train/n_test in mind (this is exactly why those columns exist per row, contracts §5).
#
# EXPECTED WALL TIME: no direct Phase 22 measurement is possible from this worktree (see the
# no-op-stub note above -- this run pays none of the real per-request/per-impression logging cost
# yet). The closest REAL reference is the identical base config/scale/algorithm/seed run for real
# in Phase 21 (paths above): 522.9s-564.3s (~8.7-9.4 minutes), single process, Release build, Apple
# M5/AppleClang -- NOT the (unrelated, speculative, pre-integration) 9-day-horizon projection in
# scripts/run_phase20_experiment.sh's header, which cites no real measurement at all. HONEST
# CAVEAT: once package A's real per-request feature-vector serialization (candidates.csv, ~672k
# extra rows) and per-impression outcome/survey writes land, this run will cost measurably more
# than the bare-world 522.9s-564.3s baseline above -- treat **~10-35 minutes** as the honest
# planning range (same P20/P21 lesson: have the integrator record the real number the first time
# this script runs to completion against a real package A build, and update this comment).
#
# Bash on purpose, NOT zsh (see scripts/run_phase15/16/17/19/20_experiment.sh precedent): the
# word-splitting this script relies on is POSIX sh behaviour zsh does not do by default.
#
# Build the Release binaries first, e.g. from the reel-rank repo root:
#   cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DREELRANK_VDB_DIR=/path/to/vector-db
#   cmake --build build-release --target simulate train_models -j
#   SIMULATE_BIN=./build-release/apps/simulate scripts/run_phase22_logworld.sh
#
# Overridable via env or flags (a flag always wins over its env var): SIMULATE_BIN, CONFIGS_DIR,
# RESULTS_ROOT, SEED (this package's four required knobs, matching run_phase20_experiment.sh's
# convention). Two secondary knobs follow the same convention for convenience, not required by
# this package's brief: TRAIN_MODELS_BIN (default: train_models next to SIMULATE_BIN's own
# directory) and MODEL_SEED (default: 4242 -- deliberately DIFFERENT from the log-world SEED so
# the "training-split"/"model-init" streams are never confused with the world-generation seed).
#
# --train executes the two train_models invocations (temporal + user_disjoint splits) right after
# the log-world run instead of only printing them; the DEFAULT is print-only (this package
# produces tooling, not experiment output -- see this repo's Phase 22 package-C brief: "you do NOT
# run experiments"). Both splits run SEQUENTIALLY against the SAME training_log (read-only to
# train_models) into their OWN --out-dir, so there is no concurrent-write hazard to reason about.
#
# SMOKE-SCALE / PLUMBING-ONLY RUNS: there is no simulation-shrinking --smoke flag on this script,
# same reasoning as run_phase20_experiment.sh's header (the simulate binary's own --smoke is a
# fixed round-robin dataset that never sets scheduler=event_queue, so it cannot exercise
# learning_v2 at all). Two independent, composable knobs smoke-test this SCRIPT'S PLUMBING only:
#   - `--configs-dir DIR` pointed at your own tiny realism-medium-retention.json (same basename,
#     smaller simulation.users/reels/dimensions, a short horizon_seconds, scheduler=event_queue)
#     exercises the real JSON-patch logic against a fast-loading dataset.
#   - `--bin PATH` pointed at any executable (e.g. /usr/bin/true) verifies arg parsing + config
#     generation + the results echo with ZERO real simulation work done. In this mode the
#     downstream "out" line simulate normally prints never appears, so the resolved training_log
#     directory cannot be discovered -- the script prints the two train_models invocations with a
#     `<UNRESOLVED -- see log>` placeholder and skips --train execution for that reason (warned,
#     not fatal).
#
# Compare / plot / test afterward:
#   python3 scripts/phase22_report.py --eval results/phase22/models-temporal/training_eval.csv,\
#       results/phase22/models-user_disjoint/training_eval.csv \
#       --calibration-dir results/phase22/models-temporal --calibration-dir \
#       results/phase22/models-user_disjoint --out results/published/phase22
#   uv run --project scripts scripts/plot_results.py <resolved logworld dir, e.g.
#       results/phase22/logworld/hnsw_ranker-seed42-STAMP> --phase22 \
#       results/phase22/models-temporal/training_eval.csv \
#       --phase22-calibration results/phase22/models-temporal \
#       --out results/published/phase22/figures
#       (plot_results.py's main() requires at least one positional result-dir before it reaches
#       ANY --phaseNN branch, same pre-existing constraint --phase20 already has -- passing the
#       logworld run's own resolved directory here is not just a workaround: it is itself a normal
#       `simulate` output directory, so this ALSO produces the standard reward_curve.png/
#       alignment_curve.png/etc. for the log-world run. See scripts/plot_results.py's module
#       docstring for the exact Phase 22 plot functions and their input contracts.)

set -euo pipefail

SIMULATE_BIN="${SIMULATE_BIN:-build-release/apps/simulate}"
CONFIGS_DIR="${CONFIGS_DIR:-configs}"
RESULTS_ROOT="${RESULTS_ROOT:-results/phase22}"
SEED="${SEED:-42}"
TRAIN_MODELS_BIN="${TRAIN_MODELS_BIN:-}"
MODEL_SEED="${MODEL_SEED:-4242}"
DO_TRAIN=0

usage() {
    cat <<'EOF'
usage: run_phase22_logworld.sh [--bin PATH] [--configs-dir DIR] [--out DIR] [--seed N]
                                [--train-bin PATH] [--model-seed N] [--train]

  --bin PATH         simulate binary (default: build-release/apps/simulate; env SIMULATE_BIN).
                     Point this at any executable (e.g. /usr/bin/true) to smoke-test config
                     generation + script plumbing without running a real simulation.
  --configs-dir DIR  directory holding realism-medium-retention.json (default: configs;
                     env CONFIGS_DIR)
  --out DIR          results root; the log-world run writes under DIR/logworld/, generated config
                     under DIR/generated-configs/, train_models output under
                     DIR/models-<split>/ (default: results/phase22; env RESULTS_ROOT)
  --seed N           log-world master seed (default: 42; env SEED) -- the pinned identical-world
                     log per Tier-5 acceptance; changing this is a DIFFERENT log world.
  --train-bin PATH   train_models binary (default: train_models next to --bin's own directory;
                     env TRAIN_MODELS_BIN)
  --model-seed N     seed passed to both train_models invocations (default: 4242; env MODEL_SEED)
  --train            actually RUN the two train_models invocations (temporal + user_disjoint)
                     after the log-world run, instead of only printing them (default: print-only)
  -h, --help         this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin) SIMULATE_BIN="$2"; shift 2 ;;
        --configs-dir) CONFIGS_DIR="$2"; shift 2 ;;
        --out) RESULTS_ROOT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --train-bin) TRAIN_MODELS_BIN="$2"; shift 2 ;;
        --model-seed) MODEL_SEED="$2"; shift 2 ;;
        --train) DO_TRAIN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "$SIMULATE_BIN" ]]; then
    echo "ERROR: simulate binary not found or not executable at: $SIMULATE_BIN" >&2
    echo "       Build it first (see the header of this script) or pass --bin / set SIMULATE_BIN." >&2
    echo "       (For a plumbing-only smoke test with no real simulation, pass --bin /usr/bin/true.)" >&2
    exit 1
fi

if [[ -z "$TRAIN_MODELS_BIN" ]]; then
    TRAIN_MODELS_BIN="$(dirname "$SIMULATE_BIN")/train_models"
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH -- required for the JSON patch step." >&2
    exit 1
fi

BASE_CFG="$CONFIGS_DIR/realism-medium-retention.json"
if [[ ! -f "$BASE_CFG" ]]; then
    echo "ERROR: config not found: $BASE_CFG" >&2
    exit 1
fi

mkdir -p "$RESULTS_ROOT/logs" "$RESULTS_ROOT/generated-configs"

GENERATED_CFG="$RESULTS_ROOT/generated-configs/realism-medium-retention-logworld.json"
python3 - "$BASE_CFG" "$GENERATED_CFG" <<'PYEOF'
import json
import sys

with open(sys.argv[1]) as f:
    cfg = json.load(f)

# Contracts §6: turn the training-log pipeline + survey on. ONLY these five keys are ever written
# here -- dict-key assignment on whatever sub-object already exists (or a fresh {} if this base
# config has none yet, which is the case today), never a wholesale block replacement -- so every
# other learning_v2 key (log_max_rows_per_file, survey.sample_rate, survey.noise_sd) and every
# other top-level block resolve through config.cpp's defaults exactly as the base file already
# does (see this script's header for the full config.cpp key-by-key cross-check).
learning_v2 = cfg.setdefault("learning_v2", {})
learning_v2["training_log"] = True
learning_v2["log_sample_rate"] = 0.25
learning_v2["log_pool_sample_rate"] = 0.01
survey = learning_v2.setdefault("survey", {})
survey["enabled"] = True

# Contracts §6 literal text. Already the base's value (the key is entirely absent from the
# committed base config, and config.cpp defaults EvaluationConfig.ecosystemMetrics to false) --
# spelled out explicitly for documentation/no-drift clarity, same convention
# scripts/run_phase20_experiment.sh's header describes ("the latter three at their config.hpp
# defaults, spelled out explicitly"). Zero behavior change.
cfg.setdefault("evaluation", {})["ecosystem_metrics"] = False

cfg["description"] = (
    "PHASE 22 LOG WORLD (docs/design/P22-CONTRACTS.md §6): configs/realism-medium-retention.json "
    "+ learning_v2.training_log=true, log_sample_rate=0.25, log_pool_sample_rate=0.01, "
    "survey.enabled=true, evaluation.ecosystem_metrics=false (explicit, already the base default). "
    "Every other key is UNCHANGED from the base config. The ONE pinned identical-world log for "
    "Tier-5 offline model evaluation -- generated by scripts/run_phase22_logworld.sh."
)

with open(sys.argv[2], "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PYEOF

echo "=== Phase 22 log-world run ==="
echo "  binary      : $SIMULATE_BIN"
echo "  configs-dir : $CONFIGS_DIR"
echo "  base config : $BASE_CFG"
echo "  generated   : $GENERATED_CFG"
echo "  algorithm   : hnsw_ranker"
echo "  out         : $RESULTS_ROOT"
echo "  seed        : $SEED"
echo "  start       : $(date)"
echo

OUT_DIR="$RESULTS_ROOT/logworld"
LOG="$RESULTS_ROOT/logs/logworld.log"
mkdir -p "$OUT_DIR"
: > "$LOG"

echo ">>> [logworld] algorithm=hnsw_ranker config=$GENERATED_CFG seed=$SEED -> $OUT_DIR"
RC=0
"$SIMULATE_BIN" --config "$GENERATED_CFG" --algorithm hnsw_ranker --seed "$SEED" --out "$OUT_DIR" \
    >> "$LOG" 2>&1 || RC=$?
if [[ $RC -eq 0 ]]; then
    echo "<<< [logworld] done"
else
    echo "<<< [logworld] FAILED (exit $RC) -- see $LOG"
fi

# simulate prints its OWN resolved (timestamped) output directory to stdout as "  out  <dir>"
# (apps/simulate.cpp) -- the same awk extraction run_phase20_experiment.sh's summary table uses --
# so this script never has to guess or reconstruct the <algorithm>-seedN-STAMP naming itself.
RESOLVED_DIR="$(awk '/^ *out /{print $2; exit}' "$LOG" 2>/dev/null || true)"

echo
echo "=== Phase 22 log-world run: DONE ==="
echo "  finished     : $(date)"
if [[ $RC -eq 0 && -n "$RESOLVED_DIR" ]]; then
    echo "  resolved dir : $RESOLVED_DIR"
else
    echo "  resolved dir : <UNRESOLVED -- see $LOG> (expected under --bin /usr/bin/true smoke mode, "
    echo "                 or any run that failed before printing its 'out' line)"
fi
echo

if [[ $RC -eq 0 && -n "$RESOLVED_DIR" ]]; then
    TRAINING_LOG_DIR="$RESOLVED_DIR/training_log"
else
    TRAINING_LOG_DIR="<UNRESOLVED -- see $LOG>"
fi

SPLITS=(temporal user_disjoint)
declare -A SPLIT_CMD_DISPLAY
for split in "${SPLITS[@]}"; do
    SPLIT_CMD_DISPLAY[$split]="$TRAIN_MODELS_BIN --log-dir $TRAINING_LOG_DIR --out-dir $RESULTS_ROOT/models-$split --split $split --seed $MODEL_SEED --survey"
done

echo "Next step -- train_models (contracts §5), one invocation per split, all eight §4.19 targets"
echo "(default; pass --targets to subset), --survey enables the survey-satisfaction regressor:"
for split in "${SPLITS[@]}"; do
    echo "  ${SPLIT_CMD_DISPLAY[$split]}"
done
echo

TRAIN_RC=0
if [[ $DO_TRAIN -eq 1 ]]; then
    if [[ $RC -ne 0 || -z "$RESOLVED_DIR" ]]; then
        echo "SKIPPING --train: no resolved training_log directory (log-world run failed, or ran" >&2
        echo "                  under a --bin smoke stub that never wrote one)." >&2
        TRAIN_RC=1
    else
        for split in "${SPLITS[@]}"; do
            out_dir="$RESULTS_ROOT/models-$split"
            mkdir -p "$out_dir"
            train_log="$RESULTS_ROOT/logs/train_models-$split.log"
            : > "$train_log"
            echo ">>> [train_models $split] -> $out_dir"
            rc=0
            # Properly quoted (unlike the display string above, which is text-only): each argument
            # survives as its own word regardless of spaces in TRAINING_LOG_DIR/RESULTS_ROOT.
            "$TRAIN_MODELS_BIN" --log-dir "$TRAINING_LOG_DIR" --out-dir "$out_dir" --split "$split" \
                --seed "$MODEL_SEED" --survey >> "$train_log" 2>&1 || rc=$?
            if [[ $rc -eq 0 ]]; then
                echo "<<< [train_models $split] done"
            else
                echo "<<< [train_models $split] FAILED (exit $rc) -- see $train_log"
                TRAIN_RC=1
            fi
        done
    fi
fi

echo
echo "Then, once both splits' training_eval.csv exist:"
echo "  python3 scripts/phase22_report.py \\"
echo "      --eval $RESULTS_ROOT/models-temporal/training_eval.csv,$RESULTS_ROOT/models-user_disjoint/training_eval.csv \\"
echo "      --calibration-dir $RESULTS_ROOT/models-temporal --calibration-dir $RESULTS_ROOT/models-user_disjoint \\"
echo "      --out results/published/phase22"
echo
echo "  uv run --project scripts scripts/plot_results.py \\"
echo "      ${RESOLVED_DIR:-<UNRESOLVED -- see $LOG>} \\"
echo "      --phase22 $RESULTS_ROOT/models-temporal/training_eval.csv \\"
echo "      --phase22-calibration $RESULTS_ROOT/models-temporal \\"
echo "      --out results/published/phase22/figures"
echo "  (the positional dir above is REQUIRED by plot_results.py's main() before it reaches any"
echo "   --phaseNN branch, same pre-existing constraint --phase20 already has -- it doubles as the"
echo "   logworld run's own standard reward_curve.png/alignment_curve.png/etc. input)"
echo "  (repeat --phase22/--phase22-calibration with the user_disjoint paths for that split's figures;"
echo "   see scripts/plot_results.py's module docstring for the exact Phase 22 plot functions)"

if [[ $RC -ne 0 ]]; then
    exit 1
fi
if [[ $DO_TRAIN -eq 1 && $TRAIN_RC -ne 0 ]]; then
    exit 1
fi
exit 0
