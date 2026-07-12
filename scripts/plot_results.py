#!/usr/bin/env python3
"""Plot metric time series from one or more ReelRank experiment result directories.

Usage
-----
    uv run --project scripts scripts/plot_results.py <result-dir> [<result-dir> ...] \
        [--labels a,b,c] [--out DIR] [--drift-at N]

    <result-dir>   One or more experiment result directories (each containing
                   learning_curve.csv, regret_curve.csv, summary.json, config.json
                   as produced by the ReelRank simulator). Multiple directories are
                   overlaid on shared axes, one line/color per run.
    --labels       Comma-separated run labels, one per result-dir, in the same
                   order (default: each directory's basename).
    --out          Output directory for PNGs (default: "plots"). Created if
                   missing. Nothing inside a result directory is ever written to.
    --drift-at     Interactions-per-user value marking drift onset, used as a
                   fallback for runs whose summary.json has no
                   adaptation.first_drift_interaction (drift_recovery.png only).

Example:
    uv run --project scripts scripts/plot_results.py \
        results/published/phase9/complete-system \
        results/published/phase10/drift-adaptive \
        --labels baseline,adaptive --out plots/phase10

This script only reads result CSV/JSON files and writes PNGs to --out; it
contains no simulation logic (design decision D15).

Produces up to four PNGs in --out, skipping (with a one-line stderr note) any
plot whose required inputs are missing across every run:

  reward_curve.png       mean_reward_per_impression vs. interactions_per_user,
                         one line per run (from learning_curve.csv).
  alignment_curve.png    mean_estimated_hidden_cosine vs. interactions; when a
                         run's learning_curve.csv also has drifted_alignment /
                         control_alignment columns (drift-configured runs only),
                         those are added as dashed / dotted lines in the same
                         run color.
  cumulative_regret.png  cumulative_regret vs. interactions (from
                         regret_curve.csv). The x-axis is derived by mapping
                         each row's `round` through learning_curve.csv's
                         round -> interactions_per_user pairing when available;
                         when learning_curve.csv is absent/unusable for a run,
                         interactions are instead computed as
                         (round + 1) * config.json's recommendation.feed_size.
  drift_recovery.png     Only produced when at least one run's
                         learning_curve.csv has a populated drifted_mean_reward
                         column. Plots drifted_mean_reward vs. interactions per
                         run (falling back to mean_reward_per_impression for
                         runs with no drift columns, so a non-drifted baseline
                         can be overlaid for comparison); draws a vertical
                         guide at each run's drift interaction
                         (summary.json's adaptation.first_drift_interaction,
                         else --drift-at); and a per-run horizontal dashed
                         guide at 0.95 * pre_drift_reward (from
                         adaptation.pre_drift_reward, else computed as the
                         mean reward over the rows preceding that run's drift
                         interaction).

Exit status: 0 if at least one plot was written, 1 if none were.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIGSIZE = (8, 5)
DPI = 150
GRID_ALPHA = 0.3
# Colorblind-safe qualitative cycle; index i is used for run i on every plot.
COLOR_CYCLE = plt.get_cmap("tab10").colors


def warn(message: str) -> None:
    print(f"plot_results: {message}", file=sys.stderr)


@dataclass
class RunData:
    label: str
    directory: Path
    color: tuple
    learning: Optional[pd.DataFrame]
    regret: Optional[pd.DataFrame]
    summary: Optional[dict]
    config: Optional[dict]


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # malformed CSV
        warn(f"failed to read {path}: {exc}")
        return None


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception as exc:  # malformed JSON
        warn(f"failed to read {path}: {exc}")
        return None


def load_run(directory: Path, label: str, color: tuple) -> RunData:
    if not directory.is_dir():
        warn(f"{label}: result directory not found: {directory}")
        return RunData(label, directory, color, None, None, None, None)

    learning = _read_csv(directory / "learning_curve.csv")
    if learning is None:
        warn(f"{label}: learning_curve.csv missing or unreadable")

    regret = _read_csv(directory / "regret_curve.csv")
    if regret is None:
        warn(f"{label}: regret_curve.csv missing or unreadable")

    summary = _read_json(directory / "summary.json")
    if summary is None:
        warn(f"{label}: summary.json missing or unreadable")

    config = _read_json(directory / "config.json")
    if config is None:
        warn(f"{label}: config.json missing or unreadable")

    return RunData(label, directory, color, learning, regret, summary, config)


def maybe_legend(ax, **kwargs) -> None:
    """Add a legend only when more than one labeled artist is present."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(handles, labels, **kwargs)


def finish_figure(fig, ax, outpath: Path, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA)
    maybe_legend(ax, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def plot_reward_curve(runs: list[RunData], outdir: Path) -> bool:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for run in runs:
        df = run.learning
        if df is None or "interactions_per_user" not in df.columns or "mean_reward_per_impression" not in df.columns:
            continue
        ax.plot(df["interactions_per_user"], df["mean_reward_per_impression"], color=run.color, label=run.label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn("skipping reward_curve.png: no run has usable learning_curve.csv data")
        return False
    finish_figure(
        fig, ax, outdir / "reward_curve.png",
        "interactions per user", "mean reward per impression",
        "Reward per Impression vs. Interactions",
    )
    return True


def plot_alignment_curve(runs: list[RunData], outdir: Path) -> bool:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for run in runs:
        df = run.learning
        if df is None or "interactions_per_user" not in df.columns or "mean_estimated_hidden_cosine" not in df.columns:
            continue
        x = df["interactions_per_user"]
        ax.plot(x, df["mean_estimated_hidden_cosine"], color=run.color, linestyle="-", label=run.label)
        plotted += 1
        if "drifted_alignment" in df.columns:
            ax.plot(x, df["drifted_alignment"], color=run.color, linestyle="--", label=f"{run.label} (drifted)")
        if "control_alignment" in df.columns:
            ax.plot(x, df["control_alignment"], color=run.color, linestyle=":", label=f"{run.label} (control)")
    if plotted == 0:
        plt.close(fig)
        warn("skipping alignment_curve.png: no run has usable learning_curve.csv data")
        return False
    finish_figure(
        fig, ax, outdir / "alignment_curve.png",
        "interactions per user", "mean estimated-hidden cosine similarity",
        "Preference Alignment vs. Interactions",
    )
    return True


def _regret_interactions(run: RunData):
    """Return (x, y) interaction/cumulative_regret series for a run, or (None, None)."""
    regret = run.regret
    if regret is None or "round" not in regret.columns or "cumulative_regret" not in regret.columns:
        return None, None

    learning = run.learning
    if learning is not None and "round" in learning.columns and "interactions_per_user" in learning.columns:
        mapping = dict(zip(learning["round"], learning["interactions_per_user"]))
        x = regret["round"].map(mapping)
        if x.notna().all():
            return x, regret["cumulative_regret"]

    feed_size = None
    if run.config is not None:
        feed_size = run.config.get("recommendation", {}).get("feed_size")
    if feed_size:
        x = (regret["round"] + 1) * feed_size
        return x, regret["cumulative_regret"]

    return None, None


def plot_cumulative_regret(runs: list[RunData], outdir: Path) -> bool:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for run in runs:
        x, y = _regret_interactions(run)
        if x is None:
            continue
        ax.plot(x, y, color=run.color, label=run.label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn("skipping cumulative_regret.png: no run has usable regret_curve.csv data")
        return False
    finish_figure(
        fig, ax, outdir / "cumulative_regret.png",
        "interactions per user", "cumulative regret (true-affinity units)",
        "Cumulative Regret vs. Interactions",
    )
    return True


def _has_drift_data(run: RunData) -> bool:
    df = run.learning
    return df is not None and "drifted_mean_reward" in df.columns and df["drifted_mean_reward"].notna().any()


def _drift_interaction(run: RunData, cli_override: Optional[int]) -> Optional[float]:
    if run.summary is not None:
        adaptation = run.summary.get("adaptation")
        if isinstance(adaptation, dict):
            value = adaptation.get("first_drift_interaction")
            if value is not None:
                return value
    return cli_override


def _pre_drift_reward(run: RunData, drift_interaction: Optional[float], reward_col: str) -> Optional[float]:
    if run.summary is not None:
        adaptation = run.summary.get("adaptation")
        if isinstance(adaptation, dict):
            value = adaptation.get("pre_drift_reward")
            if value is not None:
                return value
    if drift_interaction is None or run.learning is None:
        return None
    df = run.learning
    if "interactions_per_user" not in df.columns or reward_col not in df.columns:
        return None
    pre = df.loc[df["interactions_per_user"] < drift_interaction, reward_col].dropna()
    if pre.empty:
        return None
    return float(pre.mean())


def plot_drift_recovery(runs: list[RunData], outdir: Path, cli_drift_at: Optional[int]) -> bool:
    if not any(_has_drift_data(run) for run in runs):
        warn("skipping drift_recovery.png: no run has drift columns (drifted_mean_reward) with data")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    drift_interactions: set = set()

    for run in runs:
        df = run.learning
        if df is None or "interactions_per_user" not in df.columns:
            continue
        if "drifted_mean_reward" in df.columns and df["drifted_mean_reward"].notna().any():
            reward_col = "drifted_mean_reward"
        elif "mean_reward_per_impression" in df.columns:
            reward_col = "mean_reward_per_impression"
        else:
            continue

        x = df["interactions_per_user"]
        ax.plot(x, df[reward_col], color=run.color, label=run.label)
        plotted += 1

        drift_interaction = _drift_interaction(run, cli_drift_at)
        if drift_interaction is not None:
            drift_interactions.add(drift_interaction)

        pre_drift_reward = _pre_drift_reward(run, drift_interaction, reward_col)
        if pre_drift_reward is not None:
            ax.axhline(0.95 * pre_drift_reward, color=run.color, linestyle="--", linewidth=1, alpha=0.6)

    if plotted == 0:
        plt.close(fig)
        warn("skipping drift_recovery.png: no run has usable reward data")
        return False

    for i, interaction in enumerate(sorted(drift_interactions)):
        ax.axvline(
            interaction, color="gray", linestyle=":", linewidth=1.5,
            label="drift start" if i == 0 else None,
        )

    finish_figure(
        fig, ax, outdir / "drift_recovery.png",
        "interactions per user", "mean reward per impression",
        "Preference-Drift Recovery",
    )
    return True


def derive_labels(labels_arg: Optional[str], dirs: list[Path]) -> list[str]:
    if labels_arg:
        parts = [p.strip() for p in labels_arg.split(",")]
        if len(parts) == len(dirs):
            return parts
        warn(
            f"--labels count ({len(parts)}) does not match number of result "
            f"dirs ({len(dirs)}); using directory names instead"
        )
    return [d.name if d.name else d.resolve().name for d in dirs]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ReelRank experiment results (reward, alignment, regret, drift recovery).",
    )
    parser.add_argument(
        "result_dirs", nargs="+", metavar="result-dir",
        help="one or more experiment result directories",
    )
    parser.add_argument(
        "--labels", default=None,
        help="comma-separated run labels, one per result-dir (default: directory basenames)",
    )
    parser.add_argument(
        "--out", default="plots",
        help="output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--drift-at", type=int, default=None, dest="drift_at",
        help="interactions_per_user marking drift onset, used when no run's "
             "summary.json has adaptation.first_drift_interaction",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    dirs = [Path(d) for d in args.result_dirs]
    labels = derive_labels(args.labels, dirs)
    colors = [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(len(dirs))]

    runs = [load_run(d, label, color) for d, label, color in zip(dirs, labels, colors)]

    written = 0
    written += plot_reward_curve(runs, outdir)
    written += plot_alignment_curve(runs, outdir)
    written += plot_cumulative_regret(runs, outdir)
    written += plot_drift_recovery(runs, outdir, args.drift_at)

    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
