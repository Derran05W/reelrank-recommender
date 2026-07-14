#!/usr/bin/env python3
"""Batch driver for the ReelRank ``simulate`` binary: run a grid of (config × algorithm) arms.

Each arm is invoked as::

    <bin> --config <config> --algorithm <algo> --seed <seed> --out <out>/<config>/<algo> [--smoke]

Arms run **sequentially** and each gets its **own --out subdirectory**. Both matter: the simulator's
experiment-id is ``<algorithm>-seed<seed>-<yyyymmdd-hhmmss>`` (second resolution, config-independent,
D12), so two arms sharing an algorithm+seed and starting in the same wall-clock second would collide
on the id — serialization plus per-arm out roots make collisions impossible.

Usage
-----
    uv run --project scripts scripts/run_experiments.py \
        --configs ../configs/small.json,../configs/medium.json \
        --algorithms hnsw,hnsw_ranker,hnsw_ranker_diversity \
        --seed 42 --out ../results/sweep [--dry-run]

    # tiny end-to-end smoke over a couple of algorithms (config is supplied by --smoke):
    uv run --project scripts scripts/run_experiments.py --smoke \
        --algorithms hnsw,hnsw_ranker --out /tmp/rr-smoke

Flags
-----
    --bin          Path to the built simulate binary (default: <repo>/build-release/apps/simulate).
    --configs      Comma-separated config JSON paths (ignored under --smoke). Default small.json.
    --algorithms   Comma-separated algorithm names. Default hnsw_ranker.
    --seed         Master seed passed to every arm (default 42).
    --out          Results root; each arm writes to <out>/<config-stem>/<algorithm>/.
    --smoke        Pass --smoke to each arm (tiny CI dataset); --configs is ignored.
    --dry-run      Print the commands without executing them.

This script only orchestrates the C++ simulator (no simulation logic in Python, D15). Exit status:
0 if every arm exited 0 (or under --dry-run), 1 if any arm failed or the binary is missing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = REPO_ROOT / "build-release" / "apps" / "simulate"
# Valid algorithm names accepted by apps/simulate.cpp (kept in sync with its --help text).
KNOWN_ALGORITHMS = {
    "random", "popularity", "exact_vector", "hnsw", "hnsw_ranker",
    "hnsw_ranker_diversity", "hnsw_ranker_exploration",
}


def warn(message: str) -> None:
    print(f"run_experiments: {message}", file=sys.stderr)


def build_command(bin_path: Path, config: str | None, algorithm: str, seed: int,
                  out_dir: Path, smoke: bool) -> list[str]:
    cmd = [str(bin_path), "--algorithm", algorithm, "--seed", str(seed), "--out", str(out_dir)]
    if config is not None:
        cmd += ["--config", config]
    if smoke:
        cmd.append("--smoke")
    return cmd


def plan_arms(configs: list[str | None], algorithms: list[str], out_root: Path) -> list[tuple]:
    """Return (config, algorithm, out_dir) tuples; out_dir isolates each arm to avoid id collisions."""
    arms = []
    for config in configs:
        cfg_stem = "smoke" if config is None else Path(config).stem
        for algorithm in algorithms:
            arms.append((config, algorithm, out_root / cfg_stem / algorithm))
    return arms


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a grid of ReelRank simulate experiments (config × algorithm).",
    )
    parser.add_argument("--bin", default=str(DEFAULT_BIN),
                        help=f"path to the simulate binary (default: {DEFAULT_BIN})")
    parser.add_argument("--configs", default="configs/small.json",
                        help="comma-separated config JSON paths (ignored under --smoke)")
    parser.add_argument("--algorithms", default="hnsw_ranker",
                        help="comma-separated algorithm names")
    parser.add_argument("--seed", type=int, default=42, help="master seed for every arm")
    parser.add_argument("--out", default="results",
                        help="results root; each arm writes to <out>/<config-stem>/<algorithm>/")
    parser.add_argument("--smoke", action="store_true",
                        help="pass --smoke to each arm (tiny dataset); --configs is ignored")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="print the commands without running them")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bin_path = Path(args.bin)

    if not args.dry_run and not bin_path.exists():
        warn(f"simulate binary not found: {bin_path} (build it, or pass --bin / --dry-run)")
        return 1

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in algorithms if a not in KNOWN_ALGORITHMS]
    if unknown:
        warn(f"unknown algorithm(s): {', '.join(unknown)} "
             f"(known: {', '.join(sorted(KNOWN_ALGORITHMS))})")
        return 1

    if args.smoke:
        configs: list[str | None] = [None]
        if args.configs != "configs/small.json":
            warn("--smoke ignores --configs (the tiny dataset is built in)")
    else:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    out_root = Path(args.out)
    arms = plan_arms(configs, algorithms, out_root)
    print(f"run_experiments: {len(arms)} arm(s), seed={args.seed}, "
          f"{'dry-run' if args.dry_run else 'sequential'}")

    failures = 0
    for config, algorithm, out_dir in arms:
        cmd = build_command(bin_path, config, algorithm, args.seed, out_dir, args.smoke)
        cfg_label = "smoke" if config is None else config
        if args.dry_run:
            print(f"  [{cfg_label} / {algorithm}] {' '.join(cmd)}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{cfg_label} / {algorithm}] -> {out_dir}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            warn(f"arm [{cfg_label} / {algorithm}] exited {result.returncode}")
            failures += 1

    if failures:
        warn(f"{failures}/{len(arms)} arm(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
