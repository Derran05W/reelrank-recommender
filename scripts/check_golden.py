#!/usr/bin/env python3
"""check_golden.py — D17 golden-baseline verification tool (Realism V2, Phase 13+).

Proves that a gates-off Release run of ReelRank reproduces the committed V1 golden baseline
(``tests/golden/v1-baseline/{small,drift-medium}/``) byte-identically. This is the standing exit
criterion repeated at every V2 phase (plan/00-DESIGN-DECISIONS-V2.md D17): "gates-off re-run at
phase HEAD reproduces the committed V1 golden baseline byte-identically (deterministic CSVs;
non-timing summary fields bit-equal; Release build)".

See tests/golden/v1-baseline/README.md for full provenance, the comparison-rule rationale, and
regeneration instructions. This docstring covers usage only.

Usage:
    check_golden.py --arm small|drift-medium --run [--repo <root>]
    check_golden.py --arm small|drift-medium --run-dir <path> [--repo <root>]

Exactly one of --run / --run-dir is required.

  --run       Executes the arm's simulate command (build-release/apps/simulate, ~6 minutes on this
              machine) into a fresh temp output root, then compares the result.
  --run-dir   Compares an EXISTING run directory instead of launching a new run. Accepts either
              the leaf run directory itself (the one directly containing summary.json) or an
              output root that contains exactly one "<algorithm>-seed42-<timestamp>" subdirectory.
  --repo      reel-rank repository root that holds tests/golden/v1-baseline/ (default: ".").

Exit status is non-zero if any compared file fails.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# --- Arm definitions -------------------------------------------------------------------------
# The exact commands are documented VERBATIM (with "<tmp>" as the --out placeholder) in
# tests/golden/v1-baseline/README.md; keep the two in sync if an arm ever changes.
ARMS: dict[str, dict[str, str]] = {
    "small": {
        "config": "configs/small.json",
        "algorithm": "hnsw_ranker_diversity",
    },
    "drift-medium": {
        "config": "configs/phase10-drift.json",
        "algorithm": "hnsw_ranker",
    },
}

# The five deterministic CSVs every arm writes (D12/D17): required to be byte-identical.
DETERMINISTIC_CSVS: list[str] = [
    "diversity_metrics.csv",
    "learning_curve.csv",
    "recommendation_metrics.csv",
    "regret_curve.csv",
    "retrieval_metrics.csv",
]

# Files an experiment run writes that are intentionally NOT compared (rationale in the README):
# latency_metrics.csv is pure timing, config.json is additive-by-design (D6 — old configs stay
# valid as new keys are added with defaults, so a later-phase run's fully-resolved config.json can
# legitimately gain keys the golden predates), metadata.json is run provenance (git SHA, host,
# timestamps) that is expected to differ on every invocation by construction.
NOT_COMPARED: list[str] = ["latency_metrics.csv", "config.json", "metadata.json"]

# --- summary.json strip list -----------------------------------------------------------------
# Every wall-clock/latency/throughput-derived key in the golden summary.json files, enumerated
# individually with a one-line rationale each (identified by inspecting
# tests/golden/v1-baseline/{small,drift-medium}/summary.json directly). Everything else in
# summary.json is a deterministic function of (config, seed) and must compare bit-equal.
SUMMARY_JSON_STRIP_PATHS: list[tuple[tuple[str, ...], str]] = [
    (
        ("experiment_id",),
        "Embeds the wall-clock capture timestamp in '<algorithm>-seed<seed>-<yyyyMMDDThhmmss>' "
        "(the run's output-directory name) — a byte-identical re-run still mints a fresh "
        "timestamp, so the id string itself can never match across runs.",
    ),
    (
        ("timing", "total_wall_seconds"),
        "Wall-clock total experiment runtime (rr::Stopwatch, D9) — varies run to run even given "
        "byte-identical simulation output.",
    ),
    (
        ("timing", "retrieval_latency_ms"),
        "Wall-clock retrieval-stage latency percentiles (p50/p95/p99/max/mean over N samples) — "
        "hardware/scheduler dependent, not a function of simulated content.",
    ),
    (
        ("timing", "ranking_latency_ms"),
        "Wall-clock ranking-stage latency percentiles — hardware/scheduler dependent.",
    ),
    (
        ("timing", "reranking_latency_ms"),
        "Wall-clock reranking-stage latency percentiles — hardware/scheduler dependent.",
    ),
    (
        ("timing", "recommend_latency_ms"),
        "Wall-clock end-to-end recommend() latency percentiles — hardware/scheduler dependent.",
    ),
]

MAX_DIFF_LINES = 20


def strip_summary(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of `doc` with every SUMMARY_JSON_STRIP_PATHS entry removed."""
    stripped = copy.deepcopy(doc)
    for path, _rationale in SUMMARY_JSON_STRIP_PATHS:
        node: Any = stripped
        for key in path[:-1]:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return stripped


class CheckResult:
    def __init__(self, name: str, ok: bool, detail: list[str] | None = None):
        self.name = name
        self.ok = ok
        self.detail = detail or []


def _excerpt(lines: list[str], limit: int = MAX_DIFF_LINES) -> list[str]:
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"... ({len(lines) - limit} more line(s) omitted)"]


def compare_csv(name: str, golden_dir: Path, run_dir: Path) -> CheckResult:
    golden_path = golden_dir / name
    run_path = run_dir / name
    if not golden_path.is_file():
        return CheckResult(name, False, [f"golden file missing: {golden_path}"])
    if not run_path.is_file():
        return CheckResult(name, False, [f"run file missing: {run_path}"])

    golden_bytes = golden_path.read_bytes()
    run_bytes = run_path.read_bytes()
    if golden_bytes == run_bytes:
        return CheckResult(name, True)

    golden_lines = golden_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    run_lines = run_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            golden_lines, run_lines, fromfile=f"golden/{name}", tofile=f"run/{name}", n=1
        )
    )
    return CheckResult(name, False, _excerpt([line.rstrip("\n") for line in diff]))


def compare_summary_json(golden_dir: Path, run_dir: Path) -> CheckResult:
    name = "summary.json"
    golden_path = golden_dir / name
    run_path = run_dir / name
    if not golden_path.is_file():
        return CheckResult(name, False, [f"golden file missing: {golden_path}"])
    if not run_path.is_file():
        return CheckResult(name, False, [f"run file missing: {run_path}"])

    try:
        golden_doc = json.loads(golden_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult(name, False, [f"golden file is not valid JSON: {exc}"])
    try:
        run_doc = json.loads(run_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult(name, False, [f"run file is not valid JSON: {exc}"])

    golden_stripped = strip_summary(golden_doc)
    run_stripped = strip_summary(run_doc)
    if golden_stripped == run_stripped:
        return CheckResult(name, True)

    golden_text = json.dumps(golden_stripped, indent=2, sort_keys=True).splitlines(keepends=True)
    run_text = json.dumps(run_stripped, indent=2, sort_keys=True).splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            [line + "\n" if not line.endswith("\n") else line for line in golden_text],
            [line + "\n" if not line.endswith("\n") else line for line in run_text],
            fromfile="golden/summary.json (stripped)",
            tofile="run/summary.json (stripped)",
            n=1,
        )
    )
    return CheckResult(name, False, _excerpt([line.rstrip("\n") for line in diff]))


def find_run_dir(root: Path, algorithm: str) -> Path:
    """Resolve `root` to the leaf run directory (the one directly containing summary.json).

    Accepts either the leaf directory itself, or an output root containing exactly one
    "<algorithm>-seed42-<timestamp>" child (what --out produces / what --run's temp root holds).
    """
    if (root / "summary.json").is_file():
        return root
    candidates = sorted(p for p in root.glob(f"{algorithm}-seed42-*") if p.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        sys.exit(
            f"check_golden: no '{algorithm}-seed42-*' run directory found under {root} "
            f"(and {root} itself has no summary.json)"
        )
    joined = ", ".join(str(c) for c in candidates)
    sys.exit(
        f"check_golden: multiple run directories match '{algorithm}-seed42-*' under {root}: "
        f"{joined} — pass the specific leaf directory via --run-dir"
    )


def run_arm(repo: Path, arm: str) -> Path:
    """Execute the arm's simulate command into a fresh temp output root; return the leaf run dir."""
    spec = ARMS[arm]
    simulate_bin = repo / "build-release" / "apps" / "simulate"
    if not simulate_bin.is_file():
        sys.exit(
            f"check_golden: {simulate_bin} not found — build the Release tree first "
            f"(cmake --build build-release), see tests/golden/v1-baseline/README.md"
        )

    tmp_root = Path(tempfile.mkdtemp(prefix=f"check_golden_{arm}_"))
    print(
        f"[check_golden] launching {arm} arm (Release build) — this takes approximately 6 "
        f"minutes on this machine. Output root: {tmp_root}"
    )
    cmd = [
        str(simulate_bin),
        "--config",
        spec["config"],
        "--algorithm",
        spec["algorithm"],
        "--out",
        str(tmp_root),
    ]
    subprocess.run(cmd, cwd=repo, check=True)
    return find_run_dir(tmp_root, spec["algorithm"])


def check_arm(repo: Path, arm: str, run_dir: Path) -> bool:
    golden_dir = repo / "tests" / "golden" / "v1-baseline" / arm
    if not golden_dir.is_dir():
        sys.exit(f"check_golden: golden directory not found: {golden_dir}")

    print(f"[check_golden] arm={arm}")
    print(f"  golden: {golden_dir}")
    print(f"  run:    {run_dir}")

    results: list[CheckResult] = [compare_csv(name, golden_dir, run_dir) for name in DETERMINISTIC_CSVS]
    results.append(compare_summary_json(golden_dir, run_dir))

    all_ok = True
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"  {status}  {result.name}")
        if not result.ok:
            all_ok = False
            for line in result.detail:
                print(f"    {line}")

    for name in NOT_COMPARED:
        print(f"  SKIP  {name}  (not compared — see README)")

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run", action="store_true", help="execute the arm into a fresh temp dir, then compare"
    )
    group.add_argument("--run-dir", metavar="PATH", help="compare an existing run directory")
    parser.add_argument("--repo", default=".", help="reel-rank repo root (default: '.')")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "tests" / "golden" / "v1-baseline").is_dir():
        sys.exit(f"check_golden: {repo} does not look like the reel-rank root (no tests/golden/v1-baseline)")

    if args.run:
        run_dir = run_arm(repo, args.arm)
    else:
        run_dir = find_run_dir(Path(args.run_dir).resolve(), ARMS[args.arm]["algorithm"])

    ok = check_arm(repo, args.arm, run_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
