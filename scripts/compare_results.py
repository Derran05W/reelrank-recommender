#!/usr/bin/env python3
"""Compare metric summaries across two or more ReelRank experiment result directories.

Reads each directory's ``summary.json`` (as written by the ReelRank simulator, TDD §26), flattens
its nested sections into dotted keys (``metrics.reward_per_impression``, ``oracle.mean_regret``,
``retrieval.recall_at_10`` …), and prints one aligned Markdown table to stdout: one row per metric,
one column per run. Missing keys (metrics were added over the phases, so early runs lack later ones)
render as an empty cell rather than an error.

Usage
-----
    uv run --project scripts scripts/compare_results.py <result-dir> [<result-dir> ...] \
        [--labels a,b,...] [--csv OUT.csv] [--notes] [--precision N]

    <result-dir>   One or more experiment result directories, each containing summary.json.
    --labels       Comma-separated column labels, one per result-dir (default: directory basenames).
    --csv          Also write the flattened comparison (metrics × runs) to this CSV path.
    --notes        Include the verbose summary.json ``notes.*`` prose (dropped by default).
    --precision    Significant-figure precision for floats in the table (default: 6).

This script only reads summary.json files and prints/writes a table; it contains no simulation
logic (design decision D15). Exit status: 0 if at least one run had a readable summary.json, 1
otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

MAX_STR = 48  # long summary.json strings (notes/provenance) are truncated to keep the table narrow


def warn(message: str) -> None:
    print(f"compare_results: {message}", file=sys.stderr)


def _flatten(obj, prefix: str, out: dict, keep_notes: bool) -> None:
    """Recursively flatten scalar leaves of a nested dict into dotted keys.

    Lists (e.g. adaptation.events) are summarized as "[N items]"; the verbose notes.* subtree is
    skipped unless keep_notes is set; long strings are truncated to MAX_STR characters.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if not keep_notes and (child == "notes" or child.startswith("notes.")):
                continue
            _flatten(value, child, out, keep_notes)
    elif isinstance(obj, list):
        out[prefix] = f"[{len(obj)} items]"
    elif isinstance(obj, bool):
        out[prefix] = "true" if obj else "false"
    elif isinstance(obj, str):
        out[prefix] = obj if len(obj) <= MAX_STR else obj[: MAX_STR - 1] + "…"
    elif obj is None:
        out[prefix] = ""
    else:  # int / float
        out[prefix] = obj


def load_summary(directory: Path, keep_notes: bool) -> Optional[dict]:
    path = directory / "summary.json"
    if not path.exists():
        warn(f"{directory}: summary.json not found")
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception as exc:
        warn(f"{directory}: failed to read summary.json: {exc}")
        return None
    flat: dict = {}
    _flatten(data, "", flat, keep_notes)
    return flat


def fmt(value, precision: int) -> str:
    if isinstance(value, float):
        # %g gives significant-figure formatting and drops trailing zeros; keeps ints readable too.
        return f"{value:.{precision}g}"
    return "" if value is None else str(value)


def build_table(labels: list[str], flats: list[Optional[dict]], precision: int):
    """Return (ordered metric keys, header labels, rows) for the runs with a readable summary."""
    ordered_keys: list[str] = []
    seen = set()
    for flat in flats:
        if flat is None:
            continue
        for key in flat:
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)
    rows = []
    for key in ordered_keys:
        cells = [fmt(flat.get(key), precision) if flat is not None else "" for flat in flats]
        rows.append((key, cells))
    return ordered_keys, labels, rows


def render_markdown(header: list[str], rows) -> str:
    columns = ["metric"] + header
    widths = [len(c) for c in columns]
    for key, cells in rows:
        widths[0] = max(widths[0], len(key))
        for i, cell in enumerate(cells):
            widths[i + 1] = max(widths[i + 1], len(cell))

    def line(cells) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(columns), "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"]
    for key, cells in rows:
        out.append(line([key] + cells))
    return "\n".join(out)


def write_csv(path: Path, keys: list[str], header: list[str], flats, precision: int) -> None:
    import csv

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric"] + header)
        for key in keys:
            writer.writerow([key] + [fmt(f.get(key), precision) if f is not None else "" for f in flats])


def derive_labels(labels_arg: Optional[str], dirs: list[Path]) -> list[str]:
    if labels_arg:
        parts = [p.strip() for p in labels_arg.split(",")]
        if len(parts) == len(dirs):
            return parts
        warn(f"--labels count ({len(parts)}) != number of dirs ({len(dirs)}); using basenames")
    return [d.name if d.name else d.resolve().name for d in dirs]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ReelRank experiment summary.json metrics as a Markdown table.",
    )
    parser.add_argument("results", nargs="+", metavar="result-dir",
                        help="experiment result directories to compare")
    parser.add_argument("--labels", default=None,
                        help="comma-separated column labels, one per result-dir")
    parser.add_argument("--csv", default=None, help="also write the comparison to this CSV path")
    parser.add_argument("--notes", action="store_true",
                        help="include the verbose summary.json notes.* prose (dropped by default)")
    parser.add_argument("--precision", type=int, default=6,
                        help="significant figures for floats (default: 6)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dirs = [Path(d) for d in args.results]
    labels = derive_labels(args.labels, dirs)
    flats = [load_summary(d, args.notes) for d in dirs]

    if not any(f is not None for f in flats):
        warn("no readable summary.json in any result directory")
        return 1

    keys, header, rows = build_table(labels, flats, args.precision)
    print(render_markdown(header, rows))

    if args.csv:
        write_csv(Path(args.csv), keys, header, flats, args.precision)
        warn(f"wrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
