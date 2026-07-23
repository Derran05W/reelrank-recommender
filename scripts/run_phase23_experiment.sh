#!/usr/bin/env bash
#
# run_phase23_experiment.sh — Phase 23 closed-loop experiment matrix (V2 TDD §4.21 + Tier-5
# acceptance [closed-loop half] + §6/§10 item 8; plan Phase 23 task 3; docs/design/P23-CONTRACTS.md
# §4 [THIS package's own spec]). Package B.
#
# NINE arms, all from the SAME base world (configs/realism-medium-retention.json + the P23 base
# patch below, SEED 42, event mode) — identical worlds/streams so every cross-arm delta is a pure
# ranking-POLICY effect (Tier-5 acceptance: closed-loop comparison on identical worlds). The base
# patch (applied to EVERY arm) is exactly the two keys the contract names:
#
#   learning_v2.training_log      = true    (the ranker trains on the in-run log; also the P22
#                                            training-log pipeline the offline re-eval reads)
#   learning_v2.survey.enabled    = true    (the survey-satisfaction axis; the survey-off arm
#                                            flips THIS one key to false)
#
# Nothing else is written into the base patch: log_sample_rate (0.25), log_pool_sample_rate (0.01),
# log_max_rows_per_file, evaluation.ecosystem_metrics (false) and every other key resolve through
# config.cpp's defaults exactly as the committed base file already does (per-key `dict`-assignment,
# never a wholesale block replacement — the same discipline scripts/run_phase20/21/22_*.sh document
# and this package inherits). This is why the hand_tuned arm is byte-identical-world to the P22
# log-world run (see the REUSE note below).
#
# ARM MATRIX (arm -> CLI algorithm + the EXACT config patch on top of the base patch above):
#
#   hand_tuned          hnsw_ranker           (base patch only — NO learned gate). THE baseline/
#                                             control for every closed-loop delta. With learned_ranker
#                                             off + training_log/survey on this is EXACTLY the Phase 22
#                                             log world (byte-identical world dynamics — see REUSE).
#   semantic            hnsw_ranker           base patch + the SEMANTIC ranking patch (below): all
#                                             tuned/learned ranking weights zeroed EXCEPT similarity
#                                             (the pure-embedding-similarity control). NO learned gate.
#   learned             hnsw_learned_ranker   base patch + learning_v2.learned_ranker=true +
#     (== w_balanced)                         value_weights = contract defaults. Survey ON. This arm
#                                             DOUBLES as the w_balanced point of the frontier sweep.
#   learned_survey_off  hnsw_learned_ranker   = learned, but learning_v2.survey.enabled=FALSE (does
#                                             explicit feedback rescue the satisfaction axis? — the
#                                             LearnedRanker marks satisfaction_available=0 and the
#                                             satisfaction term contributes 0, CONTRACTS §2).
#   w_sat_100           hnsw_learned_ranker   = learned, value_weights.{satisfaction:0.60, watch:0.0}
#   w_sat_70            hnsw_learned_ranker   = learned, value_weights.{satisfaction:0.42, watch:0.18}
#   w_watch_70          hnsw_learned_ranker   = learned, value_weights.{satisfaction:0.18, watch:0.42}
#   w_watch_100         hnsw_learned_ranker   = learned, value_weights.{satisfaction:0.0,  watch:0.60}
#   w_watch_100_noexit  hnsw_learned_ranker   = w_watch_100 + value_weights.{exit:0.0, regret:0.0}
#                                             (the PURE-ENGAGEMENT vector — exit criterion 2).
#
# Total 9 DISTINCT runs (learned IS w_balanced — the frontier's balanced midpoint is not a 10th run).
#
# --- FRONTIER value_weights provenance (CONTRACTS §1 defaults + §4 sweep rule) ---
# Contract-frozen defaults (CONTRACTS §1): watch 0.30, share 0.15, follow 0.10, satisfaction 0.30,
# exit 0.10, regret 0.05. The frontier sweep varies ONLY watch and satisfaction, holding their SUM at
# the default watch+satisfaction total (0.30+0.30 = 0.60) and every OTHER weight at its default
# (share 0.15, follow 0.10, exit 0.10, regret 0.05) — "renormalized against the default total",
# CONTRACTS §4. So the sweep redistributes the fixed 0.60 mass between the two axes:
#   w_sat_100  100% satisfaction : satisfaction 0.60, watch 0.00
#   w_sat_70    70% satisfaction : satisfaction 0.42, watch 0.18   (0.70*0.60 / 0.30*0.60)
#   w_balanced  50/50 (=learned) : satisfaction 0.30, watch 0.30
#   w_watch_70  70% watch        : satisfaction 0.18, watch 0.42
#   w_watch_100 100% watch       : satisfaction 0.00, watch 0.60
# w_watch_100_noexit is w_watch_100 with exit and regret ALSO zeroed — the pure-engagement vector
# (no penalty terms at all: value = 0.60*pWatch + 0.15*pShare + 0.10*pFollow).
#
# --- SEMANTIC patch provenance (INTERPRETED — see this package's report; flagged, not guessed) ---
# The contract asks for "the P15 semantic policy ... all learned/tuned weights zeroed except
# similarity" (CONTRACTS §4), operationalized in this package's brief as "diff
# results/published/phase15/semantic/config.json's ranking block against the base's". THAT LITERAL
# DIFF IS EMPTY: the P15 `semantic` arm's RESOLVED `ranking` block is BYTE-IDENTICAL to this base
# config's resolved `ranking` block (verified by direct computation against
# results/phase22/logworld/.../config.json, the base config resolved by the same binary). The P15
# `semantic` arm differs from the base NOT by ranking weights but by ALGORITHM: it ran
# `algorithm:"hnsw"` (pure vector-similarity retrieval, WeightedRanker never applied, so its ranking
# weights are inert), whereas P23 runs `semantic` under `hnsw_ranker` (WeightedRanker LIVE) for a
# clean same-pipeline control vs hand_tuned. Under a LIVE WeightedRanker the only way to reproduce
# the P15 similarity-only policy is CONTRACTS §4's literal prescription — zero every non-similarity
# ranking weight. So the SEMANTIC ranking patch below zeroes exactly the ten non-similarity ranking
# weights that are NON-ZERO in the base resolved ranking (leaving similarity_weight=0.50 untouched,
# so score = 0.50*similarity => pure-similarity ORDER):
#   base resolved non-similarity NON-ZERO weights (all -> 0.0):
#     quality_weight 0.10, freshness_weight 0.08, popularity_weight 0.07, trending_weight 0.08,
#     creator_affinity_weight 0.07, exploration_weight 0.05, repetition_penalty 0.15,
#     duration_match_weight 0.05, impression_penalty_weight 0.05, session_topic_weight 0.05
# (duration_match/impression_penalty/session_topic are config.cpp DEFAULTS, absent from the base
# input file — they are written explicitly here so the patched world truly zeroes them; every
# already-zero V2 feature weight [clickbait/emotional_*/visual_*/...] needs no restatement.) The
# `dict.update()` is per-key, never a wholesale block replacement.
#
# --- CONFIG-KEY CROSS-CHECK (validated against docs/design/P23-CONTRACTS.md §1, NOT config.cpp) ---
# Package A lands the config surface FIRST but MAY NOT have landed it when this script is written;
# per CONTRACTS §7 packages B/C validate key names against the CONTRACTS file and the integrator
# re-validates the generated configs at execution time. Every key this script writes is a §1
# contract string:  learning_v2.training_log, learning_v2.survey.enabled, learning_v2.learned_ranker,
# learning_v2.value_weights.{watch,share,follow,satisfaction,exit,regret}  (all CONTRACTS §1 lines
# 11-24); the `ranking.*` keys the semantic patch writes are pre-existing RankingConfig fields
# present in every resolved config in results/. Algorithm strings `hnsw_ranker` / `hnsw_learned_ranker`
# are CLI `--algorithm` args (CONTRACTS §1 line 22-24: the factory rejects hnsw_learned_ranker unless
# learning_v2.learned_ranker is on — the learned arms set BOTH). If package A's landed config.cpp
# names any of these differently, the integrator's re-validation catches it; nothing here reads
# config.cpp (it may not exist yet).
#
# --- REUSE (hand_tuned == the Phase 22 log world) ---
# With learned_ranker OFF + training_log/survey ON, hand_tuned's RESOLVED config equals the P22
# log-world run's resolved config (both resolve log_sample_rate=0.25, log_pool_sample_rate=0.01,
# ecosystem_metrics=false — the P22 script wrote them explicitly, hand_tuned lets them default to
# the SAME values), so the two produce byte-identical world dynamics (the log-sampling draws are
# pinned-hash, not RNG-stream, so they do not perturb the simulation — P22 CONTRACTS §1). If a P22
# log-world run is present under $P22_LOGWORLD_GLOB this script prints a note identifying it as the
# hand_tuned world; pass `--reuse-logworld PATH` to SKIP hand_tuned's rerun and adopt PATH as its
# result dir (validated: the reused run's resolved config must differ from hand_tuned only in
# world-neutral keys — description + the log-rate/ecosystem_metrics keys — else it errors). Default
# is to RERUN hand_tuned (a self-contained matrix).
#
# --- OFFLINE RE-EVAL (the gap-analysis input, CONTRACTS §4 gap method) ---
# After the matrix, this script PRINTS (and, with --train, RUNS) one offline re-eval per LEARNED arm:
#   train_models --log-dir <arm resolved dir>/training_log --out-dir results/phase23/offline-<arm> \
#                --split temporal --seed 4242 --survey
# Each trains fresh P22 models on THAT arm's own in-run log and writes training_eval.csv — the
# held-out AUC/RMSE the gap analysis pairs against the arm's closed-loop deltas. Single temporal
# split (the held-out log tail); --seed 4242 (MODEL_SEED, deliberately != the world SEED 42);
# --survey enables the satisfaction regressor. hand_tuned/semantic also carry a training_log but the
# gap method re-evals only the learned arms (their served policy IS the learned value function).
#
# --- CONCURRENCY / WALL TIME ---
# At most MAXPAR (default 3, CONTRACTS §4 "<=3 concurrent") simulate processes at any time; arms run
# in waves of MAXPAR (bash 3.2 safe — no `wait -n`). --sequential runs one at a time. Every reported
# number except wall-clock timing is deterministic (D8/D9), UNAFFECTED by the mode.
# REAL anchor: the P22 log-world run (= this hand_tuned world, full logging pipeline on) measured
# timing.total_wall_seconds = 380.4s (~6.3 min), Apple M5 / Release / single-threaded (D13). The two
# NON-learned arms (hand_tuned, semantic) should land near that. The seven LEARNED arms additionally
# pay in-loop retraining (deterministic SGD on the accumulating in-memory shown-row matrix every
# retrain_every_hours=24 simulated hours over the 9-day horizon => ~9 retrains, cost measured OUTSIDE
# simulated time, CONTRACTS §3) plus per-request LearnedRanker scoring, so they cost MORE. Treat
# ~15-40 min/arm (Release, up to 3-concurrent) as the honest planning range (the higher end for the
# learned arms); have the integrator record the real number the first time this runs against a real
# package A build and update this comment (same P20-P22 lesson).
#
# Bash on purpose, NOT zsh (run_phase15..22 precedent): the word-splitting is POSIX sh behaviour zsh
# does not do by default. Requires bash 4+ for associative arrays (`declare -A`) -- same requirement
# as scripts/run_phase20_experiment.sh and run_phase22_logworld.sh; the `#!/usr/bin/env bash` shebang
# resolves to a modern bash on PATH (NOT macOS's /bin/bash 3.2 -- invoke as `bash scripts/...` or
# `./scripts/...`, not `/bin/bash scripts/...`). No `wait -n` is used (waves join the whole batch).
# For a long batch, run detached:
#   nohup bash scripts/run_phase23_experiment.sh > results/phase23/nohup.log 2>&1 &
#
# Build the Release binaries first, e.g. from the reel-rank repo root:
#   cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DREELRANK_VDB_DIR=/path/to/vector-db
#   cmake --build build-release --target simulate train_models -j
#   SIMULATE_BIN=./build-release/apps/simulate scripts/run_phase23_experiment.sh
#
# Overridable via env or flags (a flag always wins over its env var): SIMULATE_BIN, CONFIGS_DIR,
# RESULTS_ROOT, SEED (the four required knobs). Secondary: MAXPAR, TRAIN_MODELS_BIN, MODEL_SEED,
# ARMS, P22_LOGWORLD_GLOB.
#
# SMOKE-SCALE / PLUMBING-ONLY: no simulation-shrinking flag (the binary's own --smoke is a fixed
# round-robin dataset that never sets scheduler=event_queue, so it cannot exercise learning_v2). Two
# composable knobs smoke-test THIS SCRIPT'S plumbing: `--configs-dir DIR` at your own tiny
# realism-medium-retention.json (same basename, smaller sim.users/reels/dims, short horizon,
# scheduler=event_queue) exercises the real JSON-patch logic; `--bin /usr/bin/true` verifies arg
# parsing + generation of all 9 configs + the arm matrix with ZERO real simulation (the "generate
# configs only" smoke the exit criteria asks for — all 9 configs still land under
# $RESULTS_ROOT/generated-configs/ for a python diff-check).
#
# Compare / plot / test afterward (package B's gap analysis, then package C's tooling):
#   # 1. run the 7 offline re-evals (this script prints them; or pass --train to run them inline)
#   # 2. gap analysis (package B):
#   python3 scripts/phase23_gap_analysis.py \
#       --arm learned=<dir> --arm w_sat_100=<dir> ... \
#       --offline learned=results/phase23/offline-learned/training_eval.csv ... \
#       --baseline hand_tuned --out results/published/phase23
#   # 3. comparison + frontier/gap plots (package C): scripts/phase23_comparison.py + plot_results.py

set -euo pipefail

SIMULATE_BIN="${SIMULATE_BIN:-build-release/apps/simulate}"
CONFIGS_DIR="${CONFIGS_DIR:-configs}"
RESULTS_ROOT="${RESULTS_ROOT:-results/phase23}"
SEED="${SEED:-42}"
MAXPAR="${MAXPAR:-3}"
TRAIN_MODELS_BIN="${TRAIN_MODELS_BIN:-}"
MODEL_SEED="${MODEL_SEED:-4242}"
ARMS="${ARMS:-}"
P22_LOGWORLD_GLOB="${P22_LOGWORLD_GLOB:-results/phase22/logworld/*}"
REUSE_LOGWORLD=""
SEQUENTIAL=0
DO_TRAIN=0

# The nine arm names, in matrix order. learned == w_balanced (no separate w_balanced run).
ALL_ARMS="hand_tuned semantic learned learned_survey_off w_sat_100 w_sat_70 w_watch_70 w_watch_100 w_watch_100_noexit"
# The seven learned arms (offline re-eval targets; also the hnsw_learned_ranker algorithm set).
LEARNED_ARMS="learned learned_survey_off w_sat_100 w_sat_70 w_watch_70 w_watch_100 w_watch_100_noexit"

usage() {
    cat <<'EOF'
usage: run_phase23_experiment.sh [--bin PATH] [--configs-dir DIR] [--out DIR] [--seed N]
                                 [--max-par N] [--sequential] [--arms "a b ..."]
                                 [--reuse-logworld PATH] [--train]
                                 [--train-bin PATH] [--model-seed N]

  --bin PATH           simulate binary (default: build-release/apps/simulate; env SIMULATE_BIN).
                       Point at /usr/bin/true to smoke-test config generation + the arm matrix with
                       no real simulation (all 9 configs still land under DIR/generated-configs/).
  --configs-dir DIR    directory holding realism-medium-retention.json (default: configs;
                       env CONFIGS_DIR).
  --out DIR            results root; each arm writes DIR/<arm>/, generated per-arm configs under
                       DIR/generated-configs/, logs under DIR/logs/ (default: results/phase23;
                       env RESULTS_ROOT).
  --seed N            master world seed, applied to every arm (default: 42; env SEED) — the pinned
                       identical world; a different seed is a DIFFERENT world.
  --max-par N          max concurrent simulate processes (default: 3; env MAXPAR). CONTRACTS §4: <=3.
  --sequential         run arms one at a time instead of in waves of --max-par.
  --arms "a b ..."     subset of arms to RUN (default: all 9; env ARMS). All 9 configs are ALWAYS
                       generated; this only filters which arms are executed (re-run one failed arm).
  --reuse-logworld PATH  adopt an existing Phase 22 log-world run dir as the hand_tuned result
                       instead of rerunning it (its resolved config must differ from hand_tuned only
                       in world-neutral keys, else this errors). Default: rerun hand_tuned.
  --train              actually RUN the 7 per-learned-arm offline re-evals after the matrix, instead
                       of only printing them (default: print-only — this package produces tooling).
  --train-bin PATH     train_models binary (default: train_models next to --bin's dir; env
                       TRAIN_MODELS_BIN).
  --model-seed N       seed for the offline re-evals (default: 4242; env MODEL_SEED — != world SEED).
  -h, --help           this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin) SIMULATE_BIN="$2"; shift 2 ;;
        --configs-dir) CONFIGS_DIR="$2"; shift 2 ;;
        --out) RESULTS_ROOT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-par) MAXPAR="$2"; shift 2 ;;
        --sequential) SEQUENTIAL=1; shift ;;
        --arms) ARMS="$2"; shift 2 ;;
        --reuse-logworld) REUSE_LOGWORLD="$2"; shift 2 ;;
        --train) DO_TRAIN=1; shift ;;
        --train-bin) TRAIN_MODELS_BIN="$2"; shift 2 ;;
        --model-seed) MODEL_SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "$SIMULATE_BIN" ]]; then
    echo "ERROR: simulate binary not found or not executable at: $SIMULATE_BIN" >&2
    echo "       Build it first (see this script's header) or pass --bin / set SIMULATE_BIN." >&2
    echo "       (For a plumbing-only smoke test with no real simulation, pass --bin /usr/bin/true.)" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH -- required for the per-arm JSON patch step." >&2
    exit 1
fi
if [[ -z "$TRAIN_MODELS_BIN" ]]; then
    TRAIN_MODELS_BIN="$(dirname "$SIMULATE_BIN")/train_models"
fi

BASE_CFG="$CONFIGS_DIR/realism-medium-retention.json"
if [[ ! -f "$BASE_CFG" ]]; then
    echo "ERROR: config not found: $BASE_CFG" >&2
    exit 1
fi

mkdir -p "$RESULTS_ROOT/logs" "$RESULTS_ROOT/generated-configs"

# ---------------------------------------------------------------------------------------------------
# Per-arm value_weights vectors (CONTRACTS §1 defaults + §4 frontier sweep; see header for the rule).
# Written EXPLICITLY (full 6-key block) on every learned arm so (a) the arm uses the contract-frozen
# weights regardless of any config.cpp default drift, and (b) the generate-smoke diff between arms is
# exactly the intended 2 (or 4) keys.
# ---------------------------------------------------------------------------------------------------
VW_BALANCED='{"watch": 0.30, "share": 0.15, "follow": 0.10, "satisfaction": 0.30, "exit": 0.10, "regret": 0.05}'
VW_SAT100='{"watch": 0.00, "share": 0.15, "follow": 0.10, "satisfaction": 0.60, "exit": 0.10, "regret": 0.05}'
VW_SAT70='{"watch": 0.18, "share": 0.15, "follow": 0.10, "satisfaction": 0.42, "exit": 0.10, "regret": 0.05}'
VW_WATCH70='{"watch": 0.42, "share": 0.15, "follow": 0.10, "satisfaction": 0.18, "exit": 0.10, "regret": 0.05}'
VW_WATCH100='{"watch": 0.60, "share": 0.15, "follow": 0.10, "satisfaction": 0.00, "exit": 0.10, "regret": 0.05}'
VW_WATCH100_NOEXIT='{"watch": 0.60, "share": 0.15, "follow": 0.10, "satisfaction": 0.00, "exit": 0.00, "regret": 0.00}'

# The semantic ranking patch (see header for the full provenance): zero every non-similarity ranking
# weight that is non-zero in the base resolved ranking; similarity_weight (0.50) left untouched.
SEMANTIC_PATCH='{"quality_weight": 0.0, "freshness_weight": 0.0, "popularity_weight": 0.0, "trending_weight": 0.0, "creator_affinity_weight": 0.0, "exploration_weight": 0.0, "repetition_penalty": 0.0, "duration_match_weight": 0.0, "impression_penalty_weight": 0.0, "session_topic_weight": 0.0}'

# arm_algo <arm>            -> the --algorithm CLI string for that arm.
arm_algo() {
    case "$1" in
        hand_tuned|semantic) echo "hnsw_ranker" ;;
        *) echo "hnsw_learned_ranker" ;;
    esac
}
# arm_learned <arm>         -> 1 if this arm sets learning_v2.learned_ranker=true, else 0.
arm_learned() {
    case "$1" in hand_tuned|semantic) echo 0 ;; *) echo 1 ;; esac
}
# arm_survey <arm>          -> 1 if survey.enabled=true (default), 0 for the survey-off arm.
arm_survey() {
    case "$1" in learned_survey_off) echo 0 ;; *) echo 1 ;; esac
}
# arm_value_weights <arm>   -> the value_weights JSON for a learned arm, or "" for non-learned arms.
arm_value_weights() {
    case "$1" in
        learned|learned_survey_off) echo "$VW_BALANCED" ;;
        w_sat_100) echo "$VW_SAT100" ;;
        w_sat_70) echo "$VW_SAT70" ;;
        w_watch_70) echo "$VW_WATCH70" ;;
        w_watch_100) echo "$VW_WATCH100" ;;
        w_watch_100_noexit) echo "$VW_WATCH100_NOEXIT" ;;
        *) echo "" ;;
    esac
}
# arm_ranking_patch <arm>   -> the ranking patch JSON (semantic only), or "".
arm_ranking_patch() {
    case "$1" in semantic) echo "$SEMANTIC_PATCH" ;; *) echo "" ;; esac
}

# ---------------------------------------------------------------------------------------------------
# Generate ALL nine per-arm configs up front (fast, fail-fast). The python patcher applies the base
# patch (training_log + survey.enabled) then the arm-specific additions; never a wholesale block
# replacement, so every unlisted key resolves through config.cpp defaults identically across arms.
# ---------------------------------------------------------------------------------------------------
declare -A ARM_CFG
for arm in $ALL_ARMS; do
    out_cfg="$RESULTS_ROOT/generated-configs/realism-medium-retention-$arm.json"
    ARM_CFG[$arm]="$out_cfg"
    python3 - "$BASE_CFG" "$out_cfg" "$arm" "$(arm_learned "$arm")" "$(arm_survey "$arm")" \
        "$(arm_value_weights "$arm")" "$(arm_ranking_patch "$arm")" <<'PYEOF'
import json
import sys

base_path, out_path, arm, learned_flag, survey_flag, vw_json, rank_json = sys.argv[1:8]

with open(base_path) as f:
    cfg = json.load(f)

# --- base patch: applied to EVERY arm (CONTRACTS §4) ---
lv2 = cfg.setdefault("learning_v2", {})
lv2["training_log"] = True                     # CONTRACTS §1: learned_ranker requires training_log
survey = lv2.setdefault("survey", {})
survey["enabled"] = survey_flag == "1"         # base true; the survey-off arm flips this to false

# --- arm-specific additions (per-key assignment / dict.update, never a block replacement) ---
if learned_flag == "1":
    lv2["learned_ranker"] = True               # CONTRACTS §1: gates the hnsw_learned_ranker factory
if vw_json:
    lv2["value_weights"] = json.loads(vw_json)  # CONTRACTS §1 nested block, full 6-key vector
if rank_json:
    cfg["ranking"].update(json.loads(rank_json))  # semantic: zero non-similarity ranking weights

cfg["description"] = (
    "PHASE 23 CLOSED-LOOP ARM '%s' (docs/design/P23-CONTRACTS.md §4): "
    "configs/realism-medium-retention.json + learning_v2.training_log=true, "
    "learning_v2.survey.enabled=%s%s%s. Algorithm passed on the CLI. Every other key is UNCHANGED "
    "from the base config. Generated by scripts/run_phase23_experiment.sh (Package B)."
) % (
    arm,
    "true" if survey_flag == "1" else "false",
    ", learning_v2.learned_ranker=true" if learned_flag == "1" else "",
    (", value_weights=" + vw_json) if vw_json else (
        ", ranking similarity-only patch" if rank_json else ""),
)

with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PYEOF
done

# ---------------------------------------------------------------------------------------------------
# REUSE note + --reuse-logworld validation for the hand_tuned arm.
# ---------------------------------------------------------------------------------------------------
# World-neutral keys: differences confined to these do NOT change world dynamics (log sampling is
# pinned-hash not RNG-stream; ecosystem_metrics is evaluation-side; description is a label).
HANDTUNED_RESULT=""     # set non-empty to SKIP hand_tuned's rerun and adopt this dir
if [[ -n "$REUSE_LOGWORLD" ]]; then
    if [[ ! -d "$REUSE_LOGWORLD" ]]; then
        echo "ERROR: --reuse-logworld dir not found: $REUSE_LOGWORLD" >&2
        exit 1
    fi
    reuse_cfg="$REUSE_LOGWORLD/config.json"
    if [[ ! -f "$reuse_cfg" ]]; then
        echo "ERROR: --reuse-logworld dir has no config.json (not a simulate output dir?): $REUSE_LOGWORLD" >&2
        exit 1
    fi
    # Validate: the reused run's RESOLVED config must differ from hand_tuned's generated config only
    # in world-neutral keys. (hand_tuned's generated config is an INPUT config -- fewer keys -- so we
    # compare only the keys hand_tuned sets/inherits that matter for world dynamics; any world-key
    # mismatch is fatal.)
    if ! python3 - "${ARM_CFG[hand_tuned]}" "$reuse_cfg" <<'PYEOF'
import json, sys
gen = json.load(open(sys.argv[1]))   # hand_tuned generated (input) config
res = json.load(open(sys.argv[2]))   # reused run's resolved config.json
NEUTRAL_TOP = {"description"}
NEUTRAL_LV2 = {"log_sample_rate", "log_pool_sample_rate", "log_max_rows_per_file"}
NEUTRAL_EVAL = {"ecosystem_metrics"}
bad = []
def cmp_block(name, g, r, neutral):
    # every key hand_tuned explicitly sets must match the resolved run (world-determining),
    # excluding the world-neutral allowlist.
    for k, v in (g or {}).items():
        if k in neutral:
            continue
        if isinstance(v, dict):
            cmp_block(name + "." + k, v, (r or {}).get(k, {}), neutral)
        elif (r or {}).get(k) != v:
            bad.append("%s.%s: handtuned=%r reused=%r" % (name, k, v, (r or {}).get(k)))
for top in gen:
    if top in NEUTRAL_TOP:
        continue
    if top == "learning_v2":
        cmp_block("learning_v2", gen["learning_v2"], res.get("learning_v2", {}), NEUTRAL_LV2 | {"survey"})
        # survey.enabled must still match (world-neutral overall, but keep the axis honest)
        gs = gen["learning_v2"].get("survey", {}); rs = res.get("learning_v2", {}).get("survey", {})
        if gs.get("enabled") != rs.get("enabled"):
            bad.append("learning_v2.survey.enabled: handtuned=%r reused=%r" % (gs.get("enabled"), rs.get("enabled")))
    elif top == "evaluation":
        cmp_block("evaluation", gen["evaluation"], res.get("evaluation", {}), NEUTRAL_EVAL)
    elif isinstance(gen[top], dict):
        cmp_block(top, gen[top], res.get(top, {}), set())
    elif res.get(top) != gen[top]:
        bad.append("%s: handtuned=%r reused=%r" % (top, gen[top], res.get(top)))
if bad:
    sys.stderr.write("world-key mismatches:\n  " + "\n  ".join(bad) + "\n")
    sys.exit(1)
sys.exit(0)
PYEOF
    then
        echo "ERROR: --reuse-logworld run is NOT the hand_tuned world (world-determining keys differ" >&2
        echo "       -- see the mismatches above). Refusing to reuse it. Omit --reuse-logworld to rerun." >&2
        exit 1
    fi
    HANDTUNED_RESULT="$REUSE_LOGWORLD"
    echo "NOTE: --reuse-logworld validated: $REUSE_LOGWORLD IS the hand_tuned world (differs only in"
    echo "      world-neutral keys). Skipping hand_tuned's rerun; adopting it as the hand_tuned result."
    echo
else
    # No reuse requested: if a P22 log-world run is present, point it out (informational).
    for d in $P22_LOGWORLD_GLOB; do
        [[ -d "$d" && -f "$d/config.json" ]] || continue
        echo "NOTE: an existing Phase 22 log-world run is present: $d"
        echo "      hand_tuned reproduces this EXACT world (byte-identical dynamics: same base+seed+"
        echo "      algorithm+training_log+survey; the only config differences are the log-rate /"
        echo "      ecosystem_metrics keys at their defaults + the description). Pass"
        echo "      --reuse-logworld '$d' to skip hand_tuned's rerun and adopt it. Default: rerun."
        echo
        break
    done
fi

# ---------------------------------------------------------------------------------------------------
# Select which arms to RUN (all 9 by default; --arms/ARMS filters). Config generation above is always
# all 9 (the generate-smoke needs the full set on disk).
# ---------------------------------------------------------------------------------------------------
RUN_ARMS="$ALL_ARMS"
if [[ -n "$ARMS" ]]; then
    RUN_ARMS=""
    for a in $ARMS; do
        found=0
        for known in $ALL_ARMS; do [[ "$a" == "$known" ]] && found=1; done
        if [[ $found -eq 0 ]]; then
            echo "error: unknown arm: $a (known: $ALL_ARMS)" >&2; exit 2
        fi
        RUN_ARMS="$RUN_ARMS $a"
    done
fi
# If hand_tuned is being reused, drop it from the run list.
if [[ -n "$HANDTUNED_RESULT" ]]; then
    FILTERED=""
    for a in $RUN_ARMS; do [[ "$a" == "hand_tuned" ]] || FILTERED="$FILTERED $a"; done
    RUN_ARMS="$FILTERED"
fi

# Materialize the run list into an indexed array (order preserved).
RUN_LIST=()
for a in $RUN_ARMS; do RUN_LIST+=("$a"); done
NRUN=${#RUN_LIST[@]}

echo "=== Phase 23 closed-loop experiment matrix (Package B) ==="
echo "  binary      : $SIMULATE_BIN"
echo "  configs-dir : $CONFIGS_DIR"
echo "  base config : $BASE_CFG"
echo "  out         : $RESULTS_ROOT"
echo "  seed        : $SEED"
echo "  arms (gen)  : 9 (all configs under $RESULTS_ROOT/generated-configs/)"
printf '  arms (run)  : %d' "$NRUN"
if [[ -n "$HANDTUNED_RESULT" ]]; then printf ' (hand_tuned reused, not rerun)'; fi
echo
if [[ $SEQUENTIAL -eq 1 ]]; then
    echo "  mode        : sequential"
else
    echo "  mode        : up to $MAXPAR concurrent (waves)"
fi
echo "  start       : $(date)"
echo

# run_arm <arm> : run one arm to completion, log to its file, echo status. Never fatal (guarded).
run_arm() {
    local arm="$1"
    local algo out_dir log rc=0
    algo="$(arm_algo "$arm")"
    out_dir="$RESULTS_ROOT/$arm"
    log="$RESULTS_ROOT/logs/$arm.log"
    mkdir -p "$out_dir"; : > "$log"
    echo ">>> [$arm] algorithm=$algo config=${ARM_CFG[$arm]} seed=$SEED -> $out_dir"
    "$SIMULATE_BIN" --config "${ARM_CFG[$arm]}" --algorithm "$algo" --seed "$SEED" --out "$out_dir" \
        >> "$log" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]]; then echo "<<< [$arm] done"; else echo "<<< [$arm] FAILED (exit $rc) -- see $log"; fi
    return $rc
}

declare -A ARM_RC
if [[ $SEQUENTIAL -eq 1 ]]; then
    for arm in "${RUN_LIST[@]}"; do
        rc=0; run_arm "$arm" || rc=$?; ARM_RC[$arm]=$rc
    done
else
    for ((base = 0; base < NRUN; base += MAXPAR)); do
        pids=(); names=()
        for ((k = 0; k < MAXPAR && base + k < NRUN; k++)); do
            arm="${RUN_LIST[$((base + k))]}"
            run_arm "$arm" &
            pids+=($!); names+=("$arm")
        done
        for j in "${!pids[@]}"; do
            rc=0; wait "${pids[$j]}" || rc=$?
            ARM_RC[${names[$j]}]=$rc
        done
    done
fi

echo
echo "=== Phase 23 matrix: ALL RUN ARMS DONE ==="
echo "  finished: $(date)"
echo

# resolve_dir <arm> : echo the timestamped output dir simulate printed (awk 'out ' line), or "".
resolve_dir() {
    awk '/^ *out /{print $2; exit}' "$RESULTS_ROOT/$1.log" 2>/dev/null || true
}

# Per-arm resolved result dirs (reused hand_tuned uses its adopted dir).
declare -A ARM_DIR
overall_rc=0
printf '  %-20s %-22s %-52s %s\n' "ARM" "ALGORITHM" "RESULT DIR" "STATUS"
for arm in $ALL_ARMS; do
    algo="$(arm_algo "$arm")"
    if [[ "$arm" == "hand_tuned" && -n "$HANDTUNED_RESULT" ]]; then
        ARM_DIR[$arm]="$HANDTUNED_RESULT"; status="REUSED"
    elif [[ -z "${ARM_RC[$arm]+x}" ]]; then
        ARM_DIR[$arm]=""; status="skipped (not in --arms)"
    else
        d="$(resolve_dir "$arm")"; ARM_DIR[$arm]="$d"
        if [[ "${ARM_RC[$arm]}" -eq 0 ]]; then status="OK"; else status="FAILED (exit ${ARM_RC[$arm]})"; overall_rc=1; fi
    fi
    printf '  %-20s %-22s %-52s %s\n' "$arm" "$algo" "${ARM_DIR[$arm]:-<unresolved>}" "$status"
done
echo

# ---------------------------------------------------------------------------------------------------
# Offline re-eval commands (CONTRACTS §4 gap method) — one per LEARNED arm; PRINTED always, RUN with
# --train. Each trains fresh P22 models on the arm's own training_log for the held-out AUC/RMSE the
# gap analysis pairs against the closed-loop deltas.
# ---------------------------------------------------------------------------------------------------
echo "Next: per-learned-arm offline re-eval (CONTRACTS §4 gap method) -- held-out AUC/RMSE input to"
echo "the gap analysis. One temporal-split re-eval per learned arm on its OWN in-run training_log:"
declare -A OFFLINE_CMD
for arm in $LEARNED_ARMS; do
    d="${ARM_DIR[$arm]:-}"
    if [[ -n "$d" ]]; then tl="$d/training_log"; else tl="<UNRESOLVED -- see $RESULTS_ROOT/logs/$arm.log>"; fi
    OFFLINE_CMD[$arm]="$TRAIN_MODELS_BIN --log-dir $tl --out-dir $RESULTS_ROOT/offline-$arm --split temporal --seed $MODEL_SEED --survey"
    echo "  ${OFFLINE_CMD[$arm]}"
done
echo

TRAIN_RC=0
if [[ $DO_TRAIN -eq 1 ]]; then
    if [[ ! -x "$TRAIN_MODELS_BIN" ]]; then
        echo "SKIPPING --train: train_models binary not found/executable at: $TRAIN_MODELS_BIN" >&2
        echo "                  (pass --train-bin / set TRAIN_MODELS_BIN, or run the printed commands manually)." >&2
        TRAIN_RC=1
    else
        for arm in $LEARNED_ARMS; do
            d="${ARM_DIR[$arm]:-}"
            if [[ -z "$d" ]]; then
                echo "SKIPPING --train for $arm: no resolved result dir (arm failed, not run, or --bin smoke stub)." >&2
                TRAIN_RC=1; continue
            fi
            out_dir="$RESULTS_ROOT/offline-$arm"; mkdir -p "$out_dir"
            tlog="$RESULTS_ROOT/logs/offline-$arm.log"; : > "$tlog"
            echo ">>> [offline-$arm] -> $out_dir"
            rc=0
            "$TRAIN_MODELS_BIN" --log-dir "$d/training_log" --out-dir "$out_dir" --split temporal \
                --seed "$MODEL_SEED" --survey >> "$tlog" 2>&1 || rc=$?
            if [[ $rc -eq 0 ]]; then echo "<<< [offline-$arm] done"; else echo "<<< [offline-$arm] FAILED (exit $rc) -- see $tlog"; TRAIN_RC=1; fi
        done
    fi
    echo
fi

# ---------------------------------------------------------------------------------------------------
# Hand off to the gap analysis (package B) then package C's comparison/plots.
# ---------------------------------------------------------------------------------------------------
echo "Then the offline-vs-closed-loop gap analysis (package B), e.g.:"
echo "  python3 scripts/phase23_gap_analysis.py \\"
for arm in $LEARNED_ARMS; do
    echo "      --arm $arm=${ARM_DIR[$arm]:-<dir>} \\"
done
echo "      --arm hand_tuned=${ARM_DIR[hand_tuned]:-<dir>} --arm semantic=${ARM_DIR[semantic]:-<dir>} \\"
for arm in $LEARNED_ARMS; do
    echo "      --offline $arm=$RESULTS_ROOT/offline-$arm/training_eval.csv \\"
done
echo "      --baseline hand_tuned --out results/published/phase23"
echo
echo "Then package C's comparison + frontier/gap plots (scripts/phase23_comparison.py,"
echo "scripts/plot_results.py --phase23 ...) over the same resolved arm dirs."

if [[ $overall_rc -ne 0 ]]; then exit 1; fi
if [[ $DO_TRAIN -eq 1 && $TRAIN_RC -ne 0 ]]; then exit 1; fi
exit 0
