#!/usr/bin/env bash
#
# ReelRank demo — a guided ~10-second tour of the full recommendation pipeline.
#
#   1. Runs the complete `hnsw_ranker_diversity` pipeline (six candidate sources → weighted
#      ranker → diversity re-ranker, with online learning and ε-greedy exploration) end to end
#      on configs/small.json, and narrates a headline summary from the run's summary.json.
#   2. Uses `inspect_user --explain-user` to show ONE user's ranked feed with the per-feature
#      ranking-contribution breakdown of every item — the explainability view.
#
# Deterministic (fixed seed 42). Writes only to a scratch temp dir (never results/published/).
# Override the build location with BUILD_DIR=/path/to/build.
#
# Usage:  bash scripts/demo.sh

set -euo pipefail

# --- locate the repo root (this script lives in <repo>/scripts) ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- locate a build: BUILD_DIR override, else prefer build-release, else build ---------------
find_build() {
    if [[ -n "${BUILD_DIR:-}" ]]; then
        echo "${BUILD_DIR}"
        return 0
    fi
    for candidate in build-release build; do
        if [[ -x "${REPO_ROOT}/${candidate}/apps/simulate" ]]; then
            echo "${REPO_ROOT}/${candidate}"
            return 0
        fi
    done
    return 1
}

if ! BUILD="$(find_build)"; then
    cat >&2 <<'EOF'
error: no ReelRank build found (looked for build-release/apps/simulate then build/apps/simulate).

Build the Release configuration first:

    cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
    cmake --build build-release

...then re-run this script (or set BUILD_DIR=/path/to/build).
EOF
    exit 1
fi

SIMULATE="${BUILD}/apps/simulate"
INSPECT="${BUILD}/apps/inspect_user"
for bin in "${SIMULATE}" "${INSPECT}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "error: expected binary not found or not executable: ${bin}" >&2
        echo "       rebuild with: cmake --build ${BUILD}" >&2
        exit 1
    fi
done

CONFIG="${REPO_ROOT}/configs/small.json"
SEED=42
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/reelrank-demo.XXXXXX")"
trap 'rm -rf "${OUT_DIR}"' EXIT

echo "==================================================================================="
echo " ReelRank demo"
echo "   build:   ${BUILD}"
echo "   config:  configs/small.json  (1,000 users x 10,000 reels, seed ${SEED})"
echo "   scratch: ${OUT_DIR}"
echo "==================================================================================="
echo

# --- 1. run the full pipeline end to end -----------------------------------------------------
echo "-----------------------------------------------------------------------------------"
echo " [1/2] Simulating the complete pipeline (hnsw_ranker_diversity)"
echo "       HNSW retrieval + 5 more candidate sources -> weighted ranker (11 features)"
echo "       -> diversity re-ranker, with online learning and epsilon-greedy exploration."
echo "-----------------------------------------------------------------------------------"
"${SIMULATE}" --config "${CONFIG}" --algorithm hnsw_ranker_diversity --seed "${SEED}" \
    --out "${OUT_DIR}" >/dev/null
echo

# locate the run's summary.json (simulate writes results/<experiment-id>/ under --out)
SUMMARY="$(find "${OUT_DIR}" -name summary.json | head -1)"
if [[ -z "${SUMMARY}" ]]; then
    echo "error: simulate produced no summary.json under ${OUT_DIR}" >&2
    exit 1
fi

echo "   Headline metrics (from summary.json):"
python3 - "${SUMMARY}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m, learn, retr, div, tim = (d["metrics"], d["learning"], d["retrieval"],
                            d["diversity"], d["timing"])
rows = [
    ("reward / impression",   f"{m['reward_per_impression']:.3f}",  "engagement quality (TDD reward)"),
    ("mean true affinity",    f"{m['mean_true_affinity']:.3f}",     "hidden-preference alignment"),
    ("final est<->hidden cos",f"{learn['final_estimated_hidden_cosine']:.3f}",
                                                                    "online-learning convergence"),
    ("retrieval p95",         f"{tim['retrieval_latency_ms']['p95']:.2f} ms",
                                                                    "HNSW candidate latency"),
    ("recall@10",             f"{retr['recall_at_10']:.3f}",        "HNSW vs exact ground truth"),
    ("feed diversity",        f"{div['mean_unique_topics']:.2f} topics / "
                              f"{div['mean_unique_creators']:.2f} creators, "
                              f"repetition {div['repetition_rate']:.1f}",
                                                                    "per feed"),
]
for name, val, note in rows:
    print(f"     {name:<24} {val:<22} {note}")
PY
echo

# --- 2. explain one user's feed --------------------------------------------------------------
echo "-----------------------------------------------------------------------------------"
echo " [2/2] Explaining user 0's feed: per-feature ranking contributions"
echo "       (each item's score is the sum of its weighted feature contributions)"
echo "-----------------------------------------------------------------------------------"
EXPLAIN_JSON="${OUT_DIR}/explain-user0.json"
"${INSPECT}" --config "${CONFIG}" --seed "${SEED}" --explain-user 0 --out "${EXPLAIN_JSON}" 2>/dev/null
python3 - "${EXPLAIN_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
SOURCES = {0: "hnsw", 1: "exact", 2: "popular", 3: "trending",
           4: "fresh", 5: "creator", 6: "explore"}
feed = d["feed"]
print(f"     user {d['user_id']}, algorithm {d['algorithm']}, {len(feed)} items "
      f"(warmup {d['warmup_rounds']} rounds)\n")
for it in feed[:5]:  # first 5 items keep the tour readable
    srcs = "+".join(SOURCES.get(s, str(s)) for s in it["sources"])
    contribs = it["contributions"]
    top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    top_str = ", ".join(f"{k}={v:+.3f}" for k, v in top)
    print(f"     #{it['rank']}  reel {it['reel_id']:<6} topic {it['primary_topic']:<3} "
          f"score {it['score']:.3f}  [{srcs}]")
    print(f"          top contributions: {top_str}")
print(f"\n     (showing first 5 of {len(feed)}; every item carries all 11 feature "
      f"contributions)")
PY
echo
echo "==================================================================================="
echo " Demo complete. Scratch output was written under ${OUT_DIR} (auto-removed)."
echo "==================================================================================="
