#!/usr/bin/env bash
#
# run_phase16_experiment.sh — Phase 16 four-arm session-dynamics re-run (V2 TDD §4.6-4.9 / plan
# Phase 16 task 6: "Re-run the Phase 15 four arms with session dynamics on").
#
# Identical arm structure to scripts/run_phase15_experiment.sh, pointed at the Phase-16 "-sessions"
# config trio (the P15 configs plus "realism.session_dynamics": true — see configs/realism-medium-
# sessions{,-engagement,-proxy}.json), by default CONCURRENTLY (same collision rationale as Phase
# 15: "engagement" and "proxy" are both hnsw_ranker, so separate --out roots make same-second
# same-algorithm experiment-id collisions impossible regardless of timing):
#
#   semantic     hnsw                 configs/realism-medium-sessions.json            (D23 semantic baseline)
#   engagement   hnsw_ranker          configs/realism-medium-sessions-engagement.json (watch-correlated preset)
#   proxy        hnsw_ranker          configs/realism-medium-sessions-proxy.json      (satisfaction-proxy preset)
#   oracle       oracle_satisfaction  configs/realism-medium-sessions.json            (EVALUATION-ONLY upper
#                                                                                        bound; same
#                                                                                        arm-4 role as
#                                                                                        Phase 15)
#
# Bash on purpose, NOT zsh (see scripts/run_phase11_load.sh / run_phase15_experiment.sh precedent):
# the word-splitting this script relies on is POSIX sh behaviour zsh does not do by default. For a
# long batch, run it detached:
#   nohup bash scripts/run_phase16_experiment.sh > results/phase16/nohup.log 2>&1 &
#
# CONCURRENCY / LATENCY CAVEAT (document in any comparison written from this run's output): the
# default concurrent mode runs up to four simulate processes on one machine at once, so this run's
# `timing`/wall-clock numbers carry cache and memory-bandwidth contention (same caveat as Phase 15's
# four concurrent arms / Phase 10's seven concurrent arms). Every OTHER reported number is
# unaffected: simulation randomness is rng/clock-free (D8/D9), so reward, hidden satisfaction/
# regret, session-health aggregates, affinity, and every deterministic CSV are byte-identical to a
# solo run at the same seed. Pass --sequential for contention-free timing at the cost of roughly 4x
# wall time.
#
# SCALE AND EXPECTED WALL TIME: each medium arm is 10k users x 100k reels x 200
# interactions/user (~2M impressions), same dataset scale as Phase 15. The published Phase 15
# run's OBSERVED (concurrent, Release) wall times were semantic 124.5s (~2.1 min), oracle 145.8s
# (~2.4 min), engagement 556.0s (~9.3 min), proxy 557.7s (~9.3 min) — see
# results/published/phase15/*/summary.json's timing.total_wall_seconds — i.e. roughly 2.5-9.5
# minutes per arm, matching this script's header estimate. Phase 16 arms are EXPECTED TO RUN
# FASTER, not slower: under realism.session_dynamics, a poor-quality feed now triggers early
# probabilistic exits (V2 TDD §4.8) that truncate a user's remaining feed consumption for that
# session, so the harness does less total per-user work than the fixed interactions_per_user budget
# implies whenever exits fire — the WORSE the feed (e.g. the engagement arm's regret-laden
# sessions), the MORE truncation, so expect semantic/oracle to shrink the least and engagement/proxy
# to shrink the most relative to the Phase 15 numbers above. This is a qualitative expectation
# documented for the integrator to confirm once package A's exit mechanism is live in this script's
# tree — it is NOT re-derived here (out of scope for this package; see the statistical test's
# pending-integration notes in tests/property/session_exit_statistical_test.cpp).
#
# The oracle arm currently fails on its FIRST request in a from-scratch worktree missing package B2
# (the package-B2 stub throws std::logic_error before any dataset-scale work is wasted) — this is
# the same pre-integration caveat Phase 15's script documented; by Phase 16 package B2 should
# already be merged from Phase 15's integration, but the guard/skip-oracle path is kept identical so
# this script degrades the same way if not.
#
# Overridable via env or flags (a flag always wins over its env var): SIMULATE_BIN, CONFIGS_DIR,
# RESULTS_ROOT, SEED. Build the Release binary first, e.g. from the reel-rank repo root:
#   cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DREELRANK_VDB_DIR=/path/to/vector-db
#   cmake --build build-release --target simulate -j
#   SIMULATE_BIN=./build-release/apps/simulate scripts/run_phase16_experiment.sh
#
# SMOKE-SCALE RUNS: there is no simulation-shrinking --smoke flag on this script (the simulate
# binary's own --smoke flag is a fixed 50-user/500-reel dataset, too small to be useful for a named
# multi-arm comparison, and it does not touch the realism/ranking blocks this experiment varies). To
# smoke-test the plumbing instead, point --configs-dir at a directory holding your own tiny
# realism-medium-sessions[.json|-engagement.json|-proxy.json] triple (same basenames, smaller
# simulation.users/reels/interactions_per_user, all three realism gates on) — everything else about
# this script is unchanged.

set -euo pipefail

SIMULATE_BIN="${SIMULATE_BIN:-build-release/apps/simulate}"
CONFIGS_DIR="${CONFIGS_DIR:-configs}"
RESULTS_ROOT="${RESULTS_ROOT:-results/phase16}"
SEED="${SEED:-42}"
SEQUENTIAL=0
SKIP_ORACLE=0

usage() {
    cat <<'EOF'
usage: run_phase16_experiment.sh [--bin PATH] [--configs-dir DIR] [--out DIR] [--seed N]
                                  [--sequential] [--skip-oracle]

  --bin PATH        simulate binary (default: build-release/apps/simulate; env SIMULATE_BIN)
  --configs-dir DIR directory holding realism-medium-sessions.json /
                     realism-medium-sessions-engagement.json / realism-medium-sessions-proxy.json
                     (default: configs; env CONFIGS_DIR)
  --out DIR         results root; each arm writes under DIR/<arm-name>/ (default: results/phase16;
                     env RESULTS_ROOT)
  --seed N          master seed, applied to every arm (default: 42; env SEED)
  --sequential      run the four arms one at a time instead of concurrently (default: concurrent;
                     see the concurrency/latency caveat in this file's header)
  --skip-oracle     do not attempt the oracle_satisfaction arm
  -h, --help        this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin) SIMULATE_BIN="$2"; shift 2 ;;
        --configs-dir) CONFIGS_DIR="$2"; shift 2 ;;
        --out) RESULTS_ROOT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --sequential) SEQUENTIAL=1; shift ;;
        --skip-oracle) SKIP_ORACLE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "$SIMULATE_BIN" ]]; then
    echo "ERROR: simulate binary not found or not executable at: $SIMULATE_BIN" >&2
    echo "       Build it first (see the header of this script) or pass --bin / set SIMULATE_BIN." >&2
    exit 1
fi

SEMANTIC_CFG="$CONFIGS_DIR/realism-medium-sessions.json"
ENGAGEMENT_CFG="$CONFIGS_DIR/realism-medium-sessions-engagement.json"
PROXY_CFG="$CONFIGS_DIR/realism-medium-sessions-proxy.json"
ORACLE_CFG="$CONFIGS_DIR/realism-medium-sessions.json" # oracle ignores ranking weights; same base dataset

for f in "$SEMANTIC_CFG" "$ENGAGEMENT_CFG" "$PROXY_CFG"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: config not found: $f" >&2
        exit 1
    fi
done

mkdir -p "$RESULTS_ROOT/logs"

echo "=== Phase 16 experiment: semantic vs engagement-optimized vs satisfaction-proxy vs oracle (session dynamics on) ==="
echo "  binary      : $SIMULATE_BIN"
echo "  configs-dir : $CONFIGS_DIR"
echo "  out         : $RESULTS_ROOT"
echo "  seed        : $SEED"
if [[ $SEQUENTIAL -eq 1 ]]; then
    echo "  mode        : sequential"
else
    echo "  mode        : concurrent (see latency-contention caveat in this script's header)"
fi
echo "  oracle      : $([[ $SKIP_ORACLE -eq 1 ]] && echo "skipped (--skip-oracle)" || echo "attempted")"
echo "  start       : $(date)"
echo

# run_arm <name> <algorithm> <config-file>
#
# Runs one arm to completion, logging to $RESULTS_ROOT/logs/<name>.log, and returns simulate's exit
# code. Never uses `set -e`-fatal constructs internally so a failing arm never takes the rest of the
# script down with it — callers guard every invocation with `|| rc=$?`.
run_arm() {
    local name="$1" algorithm="$2" config="$3"
    local out_dir="$RESULTS_ROOT/$name"
    local log="$RESULTS_ROOT/logs/$name.log"
    mkdir -p "$out_dir"
    : > "$log"
    echo ">>> [$name] algorithm=$algorithm config=$config seed=$SEED -> $out_dir"
    local rc=0
    "$SIMULATE_BIN" --config "$config" --algorithm "$algorithm" --seed "$SEED" --out "$out_dir" \
        >> "$log" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "<<< [$name] done"
    else
        echo "<<< [$name] FAILED (exit $rc) -- see $log"
    fi
    return $rc
}

ARM_NAMES=(semantic engagement proxy)
ARM_ALGOS=(hnsw hnsw_ranker hnsw_ranker)
ARM_CONFIGS=("$SEMANTIC_CFG" "$ENGAGEMENT_CFG" "$PROXY_CFG")
if [[ $SKIP_ORACLE -eq 0 ]]; then
    ARM_NAMES+=(oracle)
    ARM_ALGOS+=(oracle_satisfaction)
    ARM_CONFIGS+=("$ORACLE_CFG")
fi

ARM_RC=()

if [[ $SEQUENTIAL -eq 1 ]]; then
    for i in "${!ARM_NAMES[@]}"; do
        rc=0
        run_arm "${ARM_NAMES[$i]}" "${ARM_ALGOS[$i]}" "${ARM_CONFIGS[$i]}" || rc=$?
        ARM_RC+=("$rc")
    done
else
    ARM_PIDS=()
    for i in "${!ARM_NAMES[@]}"; do
        run_arm "${ARM_NAMES[$i]}" "${ARM_ALGOS[$i]}" "${ARM_CONFIGS[$i]}" &
        ARM_PIDS+=($!)
    done
    for i in "${!ARM_NAMES[@]}"; do
        rc=0
        wait "${ARM_PIDS[$i]}" || rc=$?
        ARM_RC+=("$rc")
    done
fi

echo
echo "=== Phase 16 experiment: ALL ARMS DONE ==="
echo "  finished: $(date)"
echo
overall_rc=0
for i in "${!ARM_NAMES[@]}"; do
    name="${ARM_NAMES[$i]}"
    rc="${ARM_RC[$i]}"
    log="$RESULTS_ROOT/logs/$name.log"
    resolved_dir="$(awk '/^ *out /{print $2; exit}' "$log" 2>/dev/null || true)"
    if [[ $rc -eq 0 ]]; then
        status="OK"
    else
        status="FAILED (exit $rc)"
        overall_rc=1
    fi
    printf '  %-11s %-70s %s\n' "$name" "${resolved_dir:-$log}" "$status"
done
echo
echo "Compare the completed arms, e.g.:"
echo "  python3 scripts/phase16_comparison.py \\"
echo "      --semantic <semantic dir> --engagement <engagement dir> --proxy <proxy dir> \\"
echo "      [--oracle <oracle dir>] --out results/published/phase16"
echo
echo "Plot the headline figure(s) with the Phase 15 plotting script (unchanged; it reads the same"
echo "summary.json shape), e.g.:"
echo "  cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py \\"
echo "      ../<semantic dir> ../<engagement dir> ../<proxy dir> ../<oracle dir> \\"
echo "      --labels semantic,engagement,proxy,oracle --out ../results/published/phase16/figures"

exit $overall_rc
