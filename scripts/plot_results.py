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

Also produces Phase 11 benchmark plots when a result dir (or a parent of several) contains
retrieval_metrics.csv (from apps/benchmark_retrieval) and/or load_metrics.csv (from
apps/benchmark_recommender), found by recursive glob and concatenated:

  recall_vs_efsearch_*.png  Recall@10 vs efSearch, one line per M, one figure per
                            (vector_count, dimensions, data_distribution, ef_construction).
  recall_vs_latency_*.png   Recall@10 vs query p95 latency pareto scatter, colored by M,
                            each point annotated with its efSearch; one figure per
                            (vector_count, dimensions, data_distribution). Distance-counting
                            rows (distance_comps_per_query >= 0, inflated latency) are excluded.
  throughput_vs_threads.png recommendations/second vs client threads, one line per
                            (corpus_reels, dimensions), from load_metrics.csv.
  p99_vs_corpus.png         end-to-end and retrieval p99 vs corpus size at the max thread count
                            present, from load_metrics.csv.

Produces up to five simulation PNGs in --out, skipping (with a one-line stderr note) any
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
  engagement_vs_welfare.png  Phase 15 headline figure (V2 TDD §4.4 core
                         experiment): scatter of engagement
                         (summary.json metrics.reward_per_impression) vs.
                         mean hidden satisfaction (summary.json
                         welfare.mean_immediate_satisfaction, present only
                         under realism.latent_reactions), one labeled point
                         per run. Skipped for any run missing either value
                         (e.g. a V1/gate-off config with no `welfare` block).
  freshness_cost_frontier.png  Phase 19 headline figure (V2 TDD §4.13 core
                         experiment): scatter of cost (ranking computations
                         per request -- candidates scored; falls back to
                         summary.json counts.requests, the feed-request
                         count, when the new instrumentation is not yet
                         populated) vs. freshness (adaptation delay after
                         drift), one labeled point per run -- meant to be
                         called with one run per serving.prefetch_depth arm
                         from scripts/run_phase19_experiment.sh's four-arm
                         matrix. Both cost and freshness fields are Phase
                         19's own new instrumentation (plan Phase 19 task 2)
                         and are read by candidate summary.json key path
                         (kept in sync by hand with
                         scripts/phase19_comparison.py's CANDIDATES table);
                         skipped for any run missing both a cost and a
                         freshness figure.
  preference_divergence_hist.png  Phase 20 headline figure (V2 TDD §4.15-4.17,
                         Tier-4 acceptance item 1): histogram of per-user
                         1 - cos(sem_v_a, sem_v_b) between MATCHED user_id
                         rows of two hidden_preference_final.csv files
                         (contracts §5 frozen header) -- meant to be called
                         with the engagement-on and proxy-on arms from
                         scripts/run_phase20_experiment.sh's four-arm matrix,
                         via --phase20 ENGAGEMENT_ON_DIR PROXY_ON_DIR. Skipped
                         when either export is absent/unusable (expected
                         pre-integration, or for any -off arm -- the export
                         is gate-on only).
  retention_by_day.png   Phase 20 retention curve (contracts §6: "per-day
                         active share from longterm_metrics.csv"):
                         active_users (solid) and sessions (dashed) per
                         simulated day, one color per run, from each run's
                         longterm_metrics.csv (contracts §5 frozen header).
                         Uses the plain positional result-dir arguments (one
                         run per arm); skipped per-run when longterm_metrics.csv
                         is absent (expected for a gate-off arm or any
                         pre-Phase-20 run) and skipped entirely when no run
                         has one.
  trust_trajectory.png   Phase 20 trust trajectory (contracts §6: "mean_trust
                         by day per arm"): mean_trust per simulated day, one
                         line per run, from longterm_metrics.csv. Same input
                         contract as retention_by_day.png.
  creator_hhi_by_day.png  Phase 21 creator-concentration curve (V2 TDD §4.18,
                         docs/design/P21-CONTRACTS.md §6): creator_hhi vs.
                         simulated day, one line per run, from
                         ecosystem_metrics.csv (contracts §2 frozen header).
                         Skipped per-run when a run's ecosystem_metrics.csv is
                         absent (expected unless evaluation.ecosystem_metrics
                         is on) and skipped entirely if no run has one.
  archetype_share_by_day_<label>.png  Phase 21 archetype-mix curve (contracts
                         §6): ONE PNG PER RUN (not overlaid on shared axes,
                         unlike every other plot in this file) -- all eight
                         ecosystem_metrics.csv arch_* impression-share columns
                         (contracts §2 frozen header, catalog index order)
                         stacked against simulated day for that run alone.
                         Skipped per-run when that run's ecosystem_metrics.csv
                         is absent/unusable.
  entropy_by_day.png     Phase 21 preference-entropy trajectory (contracts
                         §3): mean_preference_entropy -- longterm_metrics.csv's
                         TRAILING column -- vs. simulated day, one line per
                         run. Reuses the same longterm_metrics.csv already
                         read for retention_by_day.png/trust_trajectory.png;
                         skipped when that column is absent (a pre-Phase-21
                         longterm_metrics.csv, or long_term not configured).
  retention_welfare_frontier.png  Phase 21 headline figure (contracts §6: "the
                         satisfaction_vs_retention headline figure"): scatter
                         of x=summary.json long_term.retention_7d vs.
                         y=welfare.mean_immediate_satisfaction, one labeled
                         point per run -- the retention-vs-hidden-welfare
                         frontier. Skipped for any run missing either value.
  calibration_<target>.png  Phase 22 reliability diagrams (V2 TDD §4.19-4.20/
                         §4.22, docs/design/P22-CONTRACTS.md §6): one PNG per
                         `calibration-<target>.csv` passed via
                         --phase22-calibration DIR (contracts §5 frozen
                         header: bin,mean_pred,mean_actual,count) -- mean_pred
                         vs. mean_actual per bin, a neutral-gray y=x "perfect
                         calibration" reference line, per-bin count encoded
                         as marker area. Target name is read from the
                         filename. Skipped per-file (warned) when absent,
                         unusable, or not matching the calibration-<target>
                         .csv naming pattern.
  offline_auc.png        Phase 22 headline figure (V2 TDD §4.19-4.20): grouped
                         bar chart of held-out AUC, one group per BINARY
                         target, one bar per model (contracts §5 frozen set:
                         learned + the three baselines -- global frequency,
                         per-source frequency, served-score-as-predictor),
                         from the training_eval.csv (contracts §5 frozen
                         header) passed via --phase22 EVAL_CSV. Rows with a
                         NaN auc are skipped first (this alone makes the
                         chart binary-targets-only, since every row of a
                         continuous target carries NaN auc by construction);
                         an AUC=0.5 "chance" guide line is drawn for
                         reference. Expects a single split's training_eval
                         .csv (a file with more than one `split` value plots
                         only the first, warned). Skipped entirely if no
                         (target, model) row has a usable auc.
  multiobjective_frontier.png  Phase 23 headline figure (V2 TDD §10 item 8,
                         docs/design/P23-CONTRACTS.md §6): scatter of x =
                         engagement (summary.json metrics.reward_per_impression)
                         vs. y = mean hidden satisfaction (welfare.
                         mean_immediate_satisfaction), marker size encoding
                         long_term.retention_7d (documented legend note; a run
                         without a usable long_term block still plots, at a
                         fixed neutral size, rather than being dropped), one
                         labeled point per run -- meant to be called with one
                         positional result-dir per arm from the Phase 23
                         nine-arm matrix (hand_tuned / semantic / learned(=
                         w_balanced) / learned_survey_off / w_sat_100 /
                         w_sat_70 / w_watch_70 / w_watch_100 /
                         w_watch_100_noexit, contracts §4). Skipped for any
                         run missing either axis value (same requirement as
                         engagement_vs_welfare.png).
  offline_closedloop_gap.png  Phase 23 offline-vs-closed-loop gap figure
                         (docs/design/P23-CONTRACTS.md §6): grouped bars, one
                         group per target (or per target+arm when more than
                         one learned arm's rows are present in the input),
                         offline held-out metric (package B's
                         `offline_learned` column) vs. closed-loop delta vs.
                         hand_tuned (package B's `aligned_cl_delta` column),
                         from package B's gap_analysis.csv passed via
                         --phase23 GAP_CSV. Column names are discovered
                         DEFENSIVELY by candidate search (package B's real
                         column names, confirmed by reading
                         scripts/phase23_gap_analysis.py directly once it
                         landed, are tried first; this function's original
                         pre-coordination guesses are kept as fallbacks) --
                         see plot_offline_closedloop_gap's docstring. Skipped
                         when the file is absent/unreadable or no target/
                         offline-value/closed-loop-delta column is found
                         under any candidate name.

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

# Canonical published figure set (--canonical): fixed retrieval grid cell + M so the committed
# figures are reproducible and comparable across the three retrieval graphs. 100k x 64d x efC200 is
# the medium §27 benchmark cell; M=16 is the only graph degree present for the clustered
# (production-like) sweep at this cell, so both distributions are shown at M=16.
CANONICAL_CELL = {"vector_count": 100000, "dimensions": 64, "ef_construction": 200}
CANONICAL_M = 16
# Fixed distribution -> color / degree -> marker maps so clustered vs random read consistently
# across the recall, latency, and Pareto retrieval figures.
DIST_COLOR = {"clustered": COLOR_CYCLE[0], "random": COLOR_CYCLE[3]}
DIST_ORDER = ["clustered", "random"]
MARKER_BY_M = {8: "s", 16: "o", 32: "^", 48: "D", 64: "v"}


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


@dataclass
class BenchmarkData:
    """Benchmark result frames for one result-dir argument (Phase 11 retrieval/load sweeps).

    Distinct from RunData (simulation curves): a benchmark result dir holds
    retrieval_metrics.csv (from apps/benchmark_retrieval) and/or load_metrics.csv (from
    apps/benchmark_recommender). Either may be absent; the dependent plots warn-skip.
    """

    label: str
    directory: Path
    retrieval: Optional[pd.DataFrame]
    load: Optional[pd.DataFrame]


def load_benchmark(directory: Path, label: str) -> BenchmarkData:
    """Load retrieval_metrics.csv / load_metrics.csv from a benchmark result dir.

    The CSVs are found by recursive glob, so `directory` may be either a single experiment-id dir
    (…/hnsw_retrieval-seed42-…/) or a parent that contains several (…/results/phase11/retrieval/);
    all matches are concatenated so multi-pass sweeps can be faceted in one call.
    """
    if not directory.is_dir():
        warn(f"{label}: result directory not found: {directory}")
        return BenchmarkData(label, directory, None, None)

    def _concat(name: str) -> Optional[pd.DataFrame]:
        frames = []
        for path in sorted(directory.rglob(name)):
            df = _read_csv(path)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    retrieval = _concat("retrieval_metrics.csv")
    load = _concat("load_metrics.csv")
    if retrieval is None and load is None:
        warn(f"{label}: no retrieval_metrics.csv or load_metrics.csv under {directory}")
    return BenchmarkData(label, directory, retrieval, load)


def _combined_retrieval(benchruns: list[BenchmarkData]) -> Optional[pd.DataFrame]:
    frames = [b.retrieval for b in benchruns if b.retrieval is not None]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # data_distribution / distance_comps_per_query are Phase 11 additions; default gracefully so
    # older (Phase 1) retrieval_metrics.csv files still plot.
    if "data_distribution" not in df.columns:
        df["data_distribution"] = "random"
    if "distance_comps_per_query" not in df.columns:
        df["distance_comps_per_query"] = -1.0
    return df


def _combined_load(benchruns: list[BenchmarkData]) -> Optional[pd.DataFrame]:
    frames = [b.load for b in benchruns if b.load is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


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


def plot_reward_curve(runs: list[RunData], outdir: Path, filename: str = "reward_curve.png") -> bool:
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
        warn(f"skipping {filename}: no run has usable learning_curve.csv data")
        return False
    finish_figure(
        fig, ax, outdir / filename,
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


def plot_engagement_vs_welfare(runs: list[RunData], outdir: Path,
                               filename: str = "engagement_vs_welfare.png") -> bool:
    """Phase 15 headline figure (V2 TDD §4.4 core experiment / plan Phase 15 task 5).

    Scatter of engagement vs. hidden welfare, one labeled point per run: x = engagement, read as
    summary.json's metrics.reward_per_impression (the same "engagement proxy" headline number used
    for every prior phase, e.g. Phase 7/10's README figures — reward_per_impression is a monotone
    function of watch ratio/seconds/like/share/follow, so it is the natural single-number engagement
    axis; mean_watch_seconds is the documented alternative if a future call site wants pure watch
    time instead). y = mean hidden satisfaction, read as summary.json's
    welfare.mean_immediate_satisfaction (Phase 14/15, D18 evaluation carve-out) -- present only for
    runs with realism.latent_reactions on, so a V1/gate-off run in the same call is skipped, not
    plotted at (x, 0). This is the plumbing that makes the phase's thesis visible: engagement and
    hidden satisfaction are DISTINCT axes (V2 TDD §3.2), so an arm can lead on one while trailing on
    the other.
    """
    usable = []
    for run in runs:
        if run.summary is None:
            continue
        engagement = run.summary.get("metrics", {}).get("reward_per_impression")
        welfare = run.summary.get("welfare")
        if engagement is None or not isinstance(welfare, dict):
            continue
        satisfaction = welfare.get("mean_immediate_satisfaction")
        if satisfaction is None:
            continue
        usable.append((run, engagement, satisfaction))

    if not usable:
        warn(f"skipping {filename}: no run has both metrics.reward_per_impression and "
             f"welfare.mean_immediate_satisfaction (welfare requires realism.latent_reactions)")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, engagement, satisfaction in usable:
        ax.scatter(engagement, satisfaction, s=110, zorder=3, color=run.color, label=run.label)
        ax.annotate(run.label, (engagement, satisfaction), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)
    ax.margins(0.18)  # headroom so corner point annotations are not clipped
    finish_figure(fig, ax, outdir / filename,
                  "engagement  (reward per impression)",
                  "mean hidden satisfaction",
                  "Engagement vs. Hidden Satisfaction")
    return True


# --- Phase 19 batch-depth frontier (V2 TDD §4.13 core experiment) ------------------------------
#
# Candidate summary.json (block, key) paths for the NEW cost/freshness instrumentation (plan Phase
# 19 task 2). Package A/B's exact key names are unknown from this worktree (grepped
# include/rr/evaluation and src/evaluation: no "ranking_computation"/"adaptation_delay" hits yet),
# so each is read by a short candidate list (variable-depth key paths), most-specific first -- kept
# in sync BY HAND with scripts/phase19_comparison.py's CANDIDATES table. The LANDED package-A schema
# nests everything under `event_mode.serving.*` (src/evaluation/results_writer.cpp); those paths
# lead each list, with the pre-merge guesses kept as fallbacks.
_P19_COST_CANDIDATES = [
    ("event_mode", "serving", "ranking_computations"),
    ("event_mode", "ranking_computation_count"),
    ("event_mode", "ranking_computations"),
    ("serving", "ranking_computation_count"),
    ("serving", "ranking_computations"),
]
_P19_FRESHNESS_CANDIDATES = [
    ("event_mode", "serving", "adaptation_delay", "mean_interactions"),
    ("event_mode", "adaptation_delay_interactions"),
    ("event_mode", "adaptation_delay_seconds"),
    ("serving", "adaptation_delay_interactions"),
    ("serving", "adaptation_delay_seconds"),
]


def _first_candidate(summary: dict, candidates: list[tuple]):
    for path in candidates:
        value = summary
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def plot_freshness_cost_frontier(runs: list[RunData], outdir: Path,
                                 filename: str = "freshness_cost_frontier.png") -> bool:
    """Phase 19 headline figure (V2 TDD §4.13 core experiment / plan Phase 19 task 3): the
    batch-depth freshness-versus-cost frontier under abrupt preference drift.

    x = cost, read as the new `ranking_computations` (candidates scored per request) instrumentation
    when present, else FALLS BACK to summary.json's already-real `counts.requests` (feed-request
    count -- a genuine, weaker cost proxy available today; see
    scripts/phase19_comparison.py's module docstring for why request count is already meaningful
    pre-integration). y = freshness, read as the new `adaptation_delay` instrumentation (interactions
    or seconds to recover pre-drift satisfaction) -- no fallback, since there is no existing
    equivalent figure to substitute (a run missing it is simply skipped, not plotted at y=0). One
    labeled point per run: call with one RunData per serving.prefetch_depth arm from
    scripts/run_phase19_experiment.sh's four-arm matrix (any run.label works, e.g. "batch1" /
    "batch3" / "batch10" / "batch20" -- this function does not parse or order by depth, it only
    plots whatever runs are passed in).
    """
    usable = []
    used_cost_fallback = False
    for run in runs:
        if run.summary is None:
            continue
        cost = _first_candidate(run.summary, _P19_COST_CANDIDATES)
        this_run_used_fallback = False
        if cost is None:
            requests = run.summary.get("counts", {}).get("requests")
            if isinstance(requests, (int, float)) and not isinstance(requests, bool):
                cost, this_run_used_fallback = requests, True
        fresh = _first_candidate(run.summary, _P19_FRESHNESS_CANDIDATES)
        if cost is None or fresh is None:
            continue
        used_cost_fallback = used_cost_fallback or this_run_used_fallback
        usable.append((run, cost, fresh))

    if not usable:
        warn(f"skipping {filename}: no run has both a cost figure (ranking_computations or, as a "
             f"fallback, counts.requests) and a freshness figure (adaptation_delay) in its "
             f"summary.json -- expected until package A lands the Phase 19 instrumentation")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, cost, fresh in usable:
        ax.scatter(cost, fresh, s=110, zorder=3, color=run.color, label=run.label)
        ax.annotate(run.label, (cost, fresh), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.margins(0.18)  # headroom so corner point annotations are not clipped
    xlabel = "cost  (ranking computations -- total candidates scored over the run)"
    if used_cost_fallback:
        xlabel += "  [fallback: feed-request count for runs without ranking_computations]"
    finish_figure(fig, ax, outdir / filename, xlabel,
                  "freshness  (adaptation delay after drift)",
                  "Batch-Depth Freshness-vs-Cost Frontier")
    return True


# --- Phase 20 policy-influence experiment (V2 TDD §4.15-4.17, Tier-4 acceptance item 1) --------
#
# These three functions are DELIBERATELY INDEPENDENT of the RunData dataclass / load_run() above
# (this package's brief: append new Phase 20 plot functions, do not alter existing functions).
# `hidden_preference_final.csv` (contracts §5, frozen schema) needs per-user matching between TWO
# SPECIFIC named arms (engagement-on vs proxy-on), which does not fit RunData's "one CSV per
# directory, one line per run" shape used by every plot above; and `longterm_metrics.csv` is a
# Phase-20-only file RunData never reads. A tiny sibling dataclass (Phase20Run) + loader covers the
# per-day plots without touching RunData/load_run at all -- both reuse the existing `_read_csv`/
# `warn`/`finish_figure`/`COLOR_CYCLE` helpers (calling them, not modifying them).

# Frozen hidden_preference_final.csv fixed columns (contracts §5); sem_v0..sem_v{D-1} follow and D
# is read from the header, never assumed. Kept in sync BY HAND with
# scripts/phase20_comparison.py's HIDDEN_PREF_REQUIRED_COLUMNS / read_hidden_preference_final --
# same defensive contract (file presence, not a guessed key; absent/unusable is never an error).
_PHASE20_HIDDEN_PREF_REQUIRED_COLUMNS = [
    "user_id", "plasticity", "churned", "sem_shift", "visual_shift", "music_shift", "emotional_shift",
]


def _read_hidden_preference_final(path: Path) -> Optional[dict]:
    """{user_id: [sem_v...]} from a frozen-schema hidden_preference_final.csv (contracts §5), or
    None if the file is absent, unreadable, or missing a required/sem_v<i> column -- never raises.
    Reuses `_read_csv` (pandas) rather than the stdlib csv module, matching this file's existing
    pandas-based style.
    """
    df = _read_csv(path)
    if df is None:
        return None
    missing = [c for c in _PHASE20_HIDDEN_PREF_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        warn(f"{path}: missing required frozen column(s) {missing}; treating as absent")
        return None
    sem_cols = sorted(
        (c for c in df.columns if c.startswith("sem_v") and c[len("sem_v"):].isdigit()),
        key=lambda c: int(c[len("sem_v"):]),
    )
    if not sem_cols:
        warn(f"{path}: no sem_v<i> columns found; treating as absent")
        return None
    rows: dict = {}
    for _, row in df.iterrows():
        uid = str(row["user_id"])
        try:
            rows[uid] = [float(row[c]) for c in sem_cols]
        except (KeyError, ValueError):
            continue
    return rows if rows else None


def _phase20_cosine(a: list, b: list) -> Optional[float]:
    if not a or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)


def plot_preference_divergence_hist(engagement_dir: Path, proxy_dir: Path, outdir: Path,
                                    engagement_label: str = "engagement",
                                    proxy_label: str = "proxy",
                                    filename: str = "preference_divergence_hist.png") -> bool:
    """Phase 20 headline figure (V2 TDD §4.15-4.17, Tier-4 acceptance item 1 / plan Phase 20 task
    5): histogram of per-user 1 - cos(sem_v_engagement, sem_v_proxy) over MATCHED user_id rows of
    two `hidden_preference_final.csv` files (contracts §5 frozen header) -- meant to be called with
    the engagement-on and proxy-on arms from scripts/run_phase20_experiment.sh's four-arm matrix
    (any two directories work; the labels are cosmetic). Takes EXPLICIT directory paths rather than
    a RunData list because this plot fundamentally needs exactly two NAMED arms matched by
    user_id, unlike every other plot in this file which overlays an arbitrary number of runs.
    Skipped (warn) when either export is absent/unusable or there are zero matched users -- the
    expected state pre-integration (packages A/B are no-op stubs in every worktree that can see
    this file) and for any `-off` arm (the export is gate-on only, contracts §5).
    """
    eng = _read_hidden_preference_final(Path(engagement_dir) / "hidden_preference_final.csv")
    proxy = _read_hidden_preference_final(Path(proxy_dir) / "hidden_preference_final.csv")
    if eng is None or proxy is None:
        warn(f"skipping {filename}: hidden_preference_final.csv missing/unusable under "
             f"{engagement_dir} and/or {proxy_dir} (expected pre-integration or for a gate-off arm)")
        return False

    distortions = []
    for uid in sorted(set(eng) & set(proxy)):
        c = _phase20_cosine(eng[uid], proxy[uid])
        if c is not None:
            distortions.append(1.0 - c)
    if not distortions:
        warn(f"skipping {filename}: zero matched (and cosine-computable) user_id rows between "
             f"{engagement_dir} and {proxy_dir}")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(distortions, bins=min(30, max(1, len(distortions))), color=COLOR_CYCLE[0],
           edgecolor="white", alpha=0.9, label=f"n={len(distortions)} matched users")
    mean_val = sum(distortions) / len(distortions)
    ax.axvline(mean_val, color=COLOR_CYCLE[3], linestyle="--", linewidth=1.5,
              label=f"mean = {mean_val:.4g}")
    finish_figure(
        fig, ax, outdir / filename,
        f"1 - cos(semantic preference): {engagement_label} vs. {proxy_label}",
        "matched users",
        f"Policy-Induced Preference Divergence ({len(distortions)} matched users)",
    )
    return True


@dataclass
class Phase20Run:
    """Minimal per-run loader for Phase 20 plots needing longterm_metrics.csv (contracts §5) --
    deliberately separate from RunData/load_run (see the section header comment above)."""

    label: str
    directory: Path
    color: tuple
    longterm: Optional[pd.DataFrame]


def _load_phase20_run(directory: Path, label: str, color: tuple) -> Phase20Run:
    longterm = _read_csv(directory / "longterm_metrics.csv")
    if longterm is None:
        warn(f"{label}: longterm_metrics.csv missing or unreadable (expected for a gate-off arm "
             f"or any pre-Phase-20 run)")
    return Phase20Run(label, directory, color, longterm)


def plot_retention_by_day(runs: list[Phase20Run], outdir: Path,
                          filename: str = "retention_by_day.png") -> bool:
    """Phase 20 retention curve (V2 TDD §4.15-4.17/§6, contracts §6: "per-day active share from
    longterm_metrics.csv"): active_users (solid line) and sessions (dashed line) per simulated day,
    one color per run, from longterm_metrics.csv's `day`/`active_users`/`sessions` columns
    (contracts §5 frozen header). Skipped (warn) for any run missing a usable longterm_metrics.csv
    -- the expected state for a gate-off arm (neither P20 gate configured) or any run predating
    Phase 20; skipped entirely (no file written) if no run has one.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for run in runs:
        df = run.longterm
        if df is None or "day" not in df.columns:
            continue
        if "active_users" in df.columns:
            ax.plot(df["day"], df["active_users"], color=run.color, linestyle="-",
                   label=f"{run.label} (active users)")
            plotted += 1
        if "sessions" in df.columns:
            ax.plot(df["day"], df["sessions"], color=run.color, linestyle="--",
                   label=f"{run.label} (sessions)")
            plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn(f"skipping {filename}: no run has a usable longterm_metrics.csv with "
             f"day/active_users/sessions columns")
        return False
    finish_figure(fig, ax, outdir / filename, "simulated day", "count",
                 "Retention: Active Users and Sessions per Simulated Day")
    return True


def plot_trust_trajectory(runs: list[Phase20Run], outdir: Path,
                          filename: str = "trust_trajectory.png") -> bool:
    """Phase 20 trust trajectory (V2 TDD §4.15-4.17/§6, contracts §6: "trust trajectory (mean_trust
    by day per arm)"): mean_trust per simulated day, one line per run, from longterm_metrics.csv's
    `day`/`mean_trust` columns (contracts §5 frozen header -- mean over ALL users at day end;
    uninitialized trust reads as the user's platformTrust trait). Skipped (warn) for any run
    missing a usable longterm_metrics.csv.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for run in runs:
        df = run.longterm
        if df is None or "day" not in df.columns or "mean_trust" not in df.columns:
            continue
        ax.plot(df["day"], df["mean_trust"], color=run.color, label=run.label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn(f"skipping {filename}: no run has a usable longterm_metrics.csv with "
             f"day/mean_trust columns")
        return False
    finish_figure(fig, ax, outdir / filename, "simulated day", "mean trust",
                 "Platform Trust Trajectory by Simulated Day")
    return True


# --- Phase 21 ecosystem failure-mode suite (V2 TDD §4.18, docs/design/P21-CONTRACTS.md §6) -----
#
# creator_hhi_by_day.png / archetype_share_by_day_<label>.png need ecosystem_metrics.csv, a
# Phase-21-only file neither RunData nor Phase20Run reads -- a tiny sibling dataclass (Phase21Run),
# mirroring the Phase 20 section's Phase20Run above, covers it without touching RunData/load_run or
# Phase20Run/_load_phase20_run at all (this package's brief: append new Phase 21 plot functions, do
# not alter existing functions/classes). entropy_by_day.png reuses Phase20Run AS-IS (it already
# reads longterm_metrics.csv in full, including the Phase 21 trailing mean_preference_entropy
# column, contracts §3) -- called with the SAME phase20_runs list main() already builds for
# plot_retention_by_day/plot_trust_trajectory. retention_welfare_frontier.png reuses RunData/
# load_run AS-IS (both fields it needs are already-loaded summary.json content) -- same shape as
# plot_engagement_vs_welfare above, called with the SAME runs list.

_PHASE21_ARCH_COLUMNS: list = [
    ("arch_genuinely_satisfying", "genuinely satisfying"),
    ("arch_useful", "useful"),
    ("arch_ragebait", "ragebait"),
    ("arch_clickbait", "clickbait"),
    ("arch_comfort", "comfort"),
    ("arch_polished_irrelevant", "polished irrelevant"),
    ("arch_niche_treasure", "niche treasure"),
    ("arch_background_music", "background music"),
]


@dataclass
class Phase21Run:
    """Minimal per-run loader for Phase 21 plots needing ecosystem_metrics.csv (contracts §2) --
    deliberately separate from RunData/load_run and from Phase20Run (see the section header comment
    above)."""

    label: str
    directory: Path
    color: tuple
    ecosystem: Optional[pd.DataFrame]


def _load_phase21_run(directory: Path, label: str, color: tuple) -> Phase21Run:
    ecosystem = _read_csv(directory / "ecosystem_metrics.csv")
    if ecosystem is None:
        warn(f"{label}: ecosystem_metrics.csv missing or unreadable (expected unless "
             f"evaluation.ecosystem_metrics is on, contracts §1, or for any pre-Phase-21 run)")
    return Phase21Run(label, directory, color, ecosystem)


def plot_creator_hhi_by_day(runs: list, outdir: Path,
                            filename: str = "creator_hhi_by_day.png") -> bool:
    """Phase 21 creator-concentration curve (V2 TDD §4.18, contracts §6: "plot_creator_hhi_by_day
    (runs, outdir) (one line per arm from ecosystem_metrics.csv)"): creator_hhi per simulated day,
    one line per run, from ecosystem_metrics.csv's day/creator_hhi columns (contracts §2 frozen
    header). Skipped (warn) for any run missing a usable ecosystem_metrics.csv -- the expected state
    for a gate-off run (evaluation.ecosystem_metrics off, contracts §1) or any pre-Phase-21 run;
    skipped entirely (no file written) if no run has one.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for run in runs:
        df = run.ecosystem
        if df is None or "day" not in df.columns or "creator_hhi" not in df.columns:
            continue
        ax.plot(df["day"], df["creator_hhi"], color=run.color, label=run.label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn(f"skipping {filename}: no run has a usable ecosystem_metrics.csv with "
             f"day/creator_hhi columns")
        return False
    finish_figure(fig, ax, outdir / filename, "simulated day", "creator HHI",
                 "Creator Concentration (HHI) by Simulated Day")
    return True


def plot_archetype_share_by_day(runs: list, outdir: Path) -> int:
    """Phase 21 archetype-mix curve (V2 TDD §4.18, contracts §6: "plot_archetype_share_by_day(runs,
    outdir) (stacked or multi-line, 8 archetypes)"): ONE PNG PER RUN (unlike every other plot in
    this file, which overlays runs on shared axes) -- each figure stacks all eight
    ecosystem_metrics.csv arch_* impression-share columns (contracts §2 frozen header, catalog index
    order) against simulated day, so one run's archetype mix over time is readable on its own (eight
    lines per run overlaid across several runs at once would be unreadable). Filenames:
    `archetype_share_by_day_<label>.png` (label sanitized to a filesystem-safe slug). Returns the
    COUNT of PNGs written -- matches the Phase 11 benchmark-plot convention below of returning a
    count rather than a bool, since this is the one plot in the simulation section that can write
    more than one file per call. Skipped per-run (warn) when that run's ecosystem_metrics.csv is
    missing or has no arch_* column; a summary warning is also emitted if NO run produced a PNG.
    """
    written = 0
    for run in runs:
        df = run.ecosystem
        if df is None or "day" not in df.columns:
            continue
        present = [(col, disp) for col, disp in _PHASE21_ARCH_COLUMNS if col in df.columns]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.stackplot(
            df["day"], *[df[col] for col, _disp in present],
            labels=[disp for _col, disp in present],
            colors=[COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(len(present))],
            alpha=0.85,
        )
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in run.label) or "run"
        filename = f"archetype_share_by_day_{slug}.png"
        finish_figure(fig, ax, outdir / filename, "simulated day", "impression share",
                     f"Archetype Mix by Simulated Day -- {run.label}")
        written += 1
    if written == 0:
        warn("skipping archetype_share_by_day: no run has a usable ecosystem_metrics.csv with "
             "day + at least one arch_* column")
    return written


def plot_entropy_by_day(runs: list, outdir: Path,
                        filename: str = "entropy_by_day.png") -> bool:
    """Phase 21 preference-entropy trajectory (V2 TDD §4.18, contracts §6: "plot_entropy_by_day
    (runs, outdir) (from longterm_metrics.csv's mean_preference_entropy trailing column)"):
    mean_preference_entropy per simulated day, one line per run, from longterm_metrics.csv's
    TRAILING column (contracts §3 -- appended last so pre-Phase-21 readers of the existing P20
    columns keep working). Takes `list[Phase20Run]` -- this file's EXISTING Phase 20 loader/
    dataclass, called but not modified, since it already reads longterm_metrics.csv in full; call
    with the SAME phase20_runs list already built for plot_retention_by_day/plot_trust_trajectory.
    Skipped (warn) for any run missing a usable longterm_metrics.csv, or whose longterm_metrics.csv
    predates the Phase 21 trailing column.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for run in runs:
        df = run.longterm
        if df is None or "day" not in df.columns or "mean_preference_entropy" not in df.columns:
            continue
        ax.plot(df["day"], df["mean_preference_entropy"], color=run.color, label=run.label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn(f"skipping {filename}: no run has a usable longterm_metrics.csv with "
             f"day/mean_preference_entropy columns (the latter is a Phase 21 addition, contracts "
             f"§3 -- absent from a pre-Phase-21 longterm_metrics.csv or a gate-off run)")
        return False
    finish_figure(fig, ax, outdir / filename, "simulated day", "mean preference entropy",
                 "Preference-Diversity Entropy by Simulated Day")
    return True


def plot_retention_welfare_frontier(runs: list, outdir: Path,
                                    filename: str = "retention_welfare_frontier.png") -> bool:
    """Phase 21 headline figure (V2 TDD §4.18, contracts §6 -- "the satisfaction_vs_retention
    headline figure"): scatter of long-term retention vs. hidden welfare, one labeled point per run:
    x = summary.json's long_term.retention_7d (P20 gate), y = welfare.mean_immediate_satisfaction
    (P14/15 gate, D18 evaluation carve-out) -- the same two axes scripts/phase21_scenarios.py's
    satisfaction_vs_retention scenario is designed around (a weight sweep between the
    engagement-optimized and satisfaction-proxy presets, contracts §7 / plan Phase 21 task 2). Takes
    `list[RunData]` -- this file's EXISTING loader/dataclass, called but not modified, matching
    plot_engagement_vs_welfare's shape exactly (both fields are already-loaded summary.json
    content). Skipped for any run missing either value (e.g. a run with neither P20 gate on, or
    without realism.latent_reactions).
    """
    usable = []
    for run in runs:
        if run.summary is None:
            continue
        long_term = run.summary.get("long_term")
        retention = long_term.get("retention_7d") if isinstance(long_term, dict) else None
        welfare = run.summary.get("welfare")
        if retention is None or not isinstance(welfare, dict):
            continue
        satisfaction = welfare.get("mean_immediate_satisfaction")
        if satisfaction is None:
            continue
        usable.append((run, retention, satisfaction))

    if not usable:
        warn(f"skipping {filename}: no run has both long_term.retention_7d and "
             f"welfare.mean_immediate_satisfaction (long_term requires a P20 gate; welfare "
             f"requires realism.latent_reactions)")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, retention, satisfaction in usable:
        ax.scatter(retention, satisfaction, s=110, zorder=3, color=run.color, label=run.label)
        ax.annotate(run.label, (retention, satisfaction), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)
    ax.margins(0.18)  # headroom so corner point annotations are not clipped
    finish_figure(fig, ax, outdir / filename, "retention (7d)", "mean hidden satisfaction",
                 "Retention vs. Hidden-Welfare Frontier")
    return True


# --- Phase 22 offline evaluation (V2 TDD §4.19-4.20/§4.22, docs/design/P22-CONTRACTS.md §6) ----
#
# plot_calibration() / plot_offline_auc() read scripts/phase22_report.py's own inputs --
# calibration-<target>.csv and training_eval.csv (contracts §5 frozen headers), produced by
# `train_models` (one invocation per split, scripts/run_phase22_logworld.sh). Both are
# DELIBERATELY INDEPENDENT of RunData/load_run and of every prior phase's loader dataclass
# (Phase20Run/Phase21Run) above: this data has no per-experiment result directory at all -- it is
# per-(target,model,split) rows in one CSV, or one small file per target -- so neither existing
# shape fits, matching this file's own established precedent (the Phase 20/21 section headers
# above) of adding a small independent function pair rather than bending an unrelated loader to
# fit. Model color is FIXED by model identity (never cycled by target), consistent with every
# other qualitative encoding in this file (COLOR_CYCLE[i % len(COLOR_CYCLE)] indexed by a STABLE
# list, exactly like load_run's per-run colors above).

_PHASE22_MODELS = ["learned", "global_frequency", "per_source_frequency", "served_score"]
_PHASE22_MODEL_LABELS = {
    "learned": "learned",
    "global_frequency": "global frequency",
    "per_source_frequency": "per-source frequency",
    "served_score": "served score",
}


def plot_calibration(calibration_csvs: list, outdir: Path) -> int:
    """Phase 22 reliability diagrams (contracts §6: "plot_calibration(target_csvs, outdir), one
    PNG per target"): mean_pred (x) vs. mean_actual (y) per bin, ONE PNG PER
    `calibration-<target>.csv` given (contracts §5 frozen header: bin,mean_pred,mean_actual,count
    -- 10 equal-count bins), plus a neutral-gray dashed y=x "perfect calibration" reference line
    and per-bin `count` encoded as marker AREA (matplotlib's scatter `s` parameter is already an
    area in points^2, so scaling it linearly with count is the perceptually-correct
    area-proportional-to-value encoding -- deliberately not a count-bar underlay, which would need
    a second y-axis and this file has no dual-axis chart anywhere). The target name is read from
    the filename (`calibration-<target>.csv`, contracts' own frozen naming -- e.g. as copied by
    scripts/phase22_report.py's --calibration-dir into
    <out>/calibration/<source-dir-slug>/calibration-<target>.csv). A file whose name does not
    match that pattern, or that is missing/unreadable/missing a required column/empty after
    dropping unusable rows, is skipped with a warning (same per-file warn-skip convention as
    plot_archetype_share_by_day above). Returns the COUNT of PNGs written (same convention as
    plot_archetype_share_by_day / the Phase 11 benchmark plots below).
    """
    written = 0
    for raw_path in calibration_csvs:
        path = Path(raw_path)
        stem = path.stem  # "calibration-<target>" with the .csv suffix stripped
        if not stem.startswith("calibration-") or len(stem) <= len("calibration-"):
            warn(f"skipping {path}: filename does not match calibration-<target>.csv")
            continue
        target = stem[len("calibration-"):]

        df = _read_csv(path)
        if df is None:
            continue
        required = {"bin", "mean_pred", "mean_actual", "count"}
        missing = required - set(df.columns)
        if missing:
            warn(f"skipping {path}: missing required column(s) {sorted(missing)}")
            continue
        df = df.dropna(subset=["bin", "mean_pred", "mean_actual", "count"])
        if df.empty:
            warn(f"skipping {path}: no usable (non-NaN) row in bin/mean_pred/mean_actual/count")
            continue
        df = df.sort_values("bin")

        x, y = df["mean_pred"].tolist(), df["mean_actual"].tolist()
        counts = df["count"].tolist()
        max_count = max(counts) if counts else 0.0
        sizes = [30.0 + 270.0 * (c / max_count) if max_count > 0 else 60.0 for c in counts]

        fig, ax = plt.subplots(figsize=FIGSIZE)
        combined = x + y
        lo, hi = min(combined), max(combined)
        pad = 0.03 * (hi - lo) if hi > lo else 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", linestyle="--",
               linewidth=1.5, zorder=1, label="perfect calibration (y=x)")
        ax.plot(x, y, color=COLOR_CYCLE[0], linewidth=1.5, zorder=2, label=target)
        ax.scatter(x, y, s=sizes, color=COLOR_CYCLE[0], zorder=3, edgecolor="white", linewidth=0.6)
        ax.text(0.02, 0.98, "marker area ∝ bin count", transform=ax.transAxes, va="top",
               fontsize=7, color="0.4")
        finish_figure(fig, ax, outdir / f"calibration_{target}.png",
                     "predicted (mean per bin)", "actual (mean per bin)",
                     f"Calibration -- {target} (n={sum(counts):g})")
        written += 1

    if written == 0:
        warn("skipping plot_calibration: no usable calibration-<target>.csv given")
    return written


def plot_offline_auc(training_eval_csv, outdir: Path, filename: str = "offline_auc.png") -> bool:
    """Phase 22 headline figure (contracts §6: "plot_offline_auc(training_eval, outdir), grouped
    bar: learned vs baselines per target"): grouped bar chart, one group per BINARY target, one
    bar per model (contracts §5 frozen model set: learned + the three baselines), height = held-out
    AUC. Expects a SINGLE split's training_eval.csv (call once per split -- matching
    `train_models`' own one-split-per-invocation output, contracts §5); if the given file DOES
    carry more than one `split` value, only the first (sorted) is plotted, with a warning, rather
    than conflating two splits' AUCs in one bar group. Rows with a NaN `auc` are skipped BEFORE
    grouping (contracts' own "NaN where inapplicable") -- since every row of a continuous target
    (e.g. watch_ratio) carries NaN auc by construction, this single filter is what makes the chart
    "binary targets only" (V2 TDD §4.19/§4.20) without a separate target-type lookup; a target
    left with fewer than four models after that filter still renders with a gap (never a
    misleading zero-height bar) for whichever model's row was NaN/absent. A horizontal AUC=0.5
    "chance" guide (neutral gray dashed, matching plot_calibration's y=x reference convention)
    gives every bar an absolute reference, not just a relative one.
    """
    df = _read_csv(Path(training_eval_csv))
    if df is None:
        warn(f"skipping {filename}: {training_eval_csv} missing or unreadable")
        return False
    required = {"target", "model", "auc"}
    missing = required - set(df.columns)
    if missing:
        warn(f"skipping {filename}: {training_eval_csv} missing required column(s) {sorted(missing)}")
        return False

    if "split" in df.columns:
        splits = sorted(s for s in df["split"].dropna().unique())
        if len(splits) > 1:
            warn(f"{training_eval_csv} contains multiple splits {splits}; plotting only "
                 f"'{splits[0]}' (call plot_offline_auc once per split's own training_eval.csv)")
            df = df[df["split"] == splits[0]]

    df = df[df["auc"].notna()]
    if df.empty:
        warn(f"skipping {filename}: no row in {training_eval_csv} has a non-NaN auc (binary "
             f"targets only -- expected if every target present is continuous, e.g. watch_ratio)")
        return False

    targets = sorted(df["target"].unique())
    n_models = len(_PHASE22_MODELS)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(FIGSIZE[0], 1.4 * len(targets)), FIGSIZE[1]))
    any_bar = False
    for i, model in enumerate(_PHASE22_MODELS):
        offset = (i - (n_models - 1) / 2.0) * width
        xs, hs = [], []
        for xi, target in enumerate(targets):
            rows = df[(df["target"] == target) & (df["model"] == model)]
            if rows.empty:
                continue
            xs.append(xi + offset)
            hs.append(float(rows["auc"].iloc[0]))
        if not hs:
            continue
        ax.bar(xs, hs, width=width, color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
              label=_PHASE22_MODEL_LABELS.get(model, model), zorder=3)
        any_bar = True

    if not any_bar:
        plt.close(fig)
        warn(f"skipping {filename}: no (target, model) pair had a usable auc value")
        return False

    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1.2, zorder=1, label="chance (AUC=0.5)")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    finish_figure(fig, ax, outdir / filename, "target", "AUC (held-out)",
                 "Offline AUC -- Learned vs. Baselines (binary targets)")
    return True


# --- Phase 23 multi-objective frontier + offline/closed-loop gap (V2 TDD §4.21/§10 item 8,
# docs/design/P23-CONTRACTS.md §6) --------------------------------------------------------------
#
# plot_multiobjective_frontier reuses the EXISTING RunData/load_run AS-IS (both fields it needs --
# metrics.reward_per_impression and welfare.mean_immediate_satisfaction -- are already-loaded
# summary.json content, plus long_term.retention_7d for marker size; same shape as
# plot_engagement_vs_welfare/plot_retention_welfare_frontier above, called but not modified, per
# this package's append-only brief). plot_offline_closedloop_gap reads package B's
# gap_analysis.csv (scripts/phase23_gap_analysis.py, contracts §4). THIS PACKAGE STARTED BEFORE
# PACKAGE B LANDED that script (contracts §4 describes its CONTENT -- "per-target table -- offline
# held-out AUC/RMSE ... vs closed-loop deltas" -- but freezes no exact column names/types the way
# e.g. contracts §3's retraining_log.csv header is frozen verbatim), so this function was written
# candidate-column-search style, the same `_first_candidate` philosophy this file's own Phase 19
# section already established for an undetermined pre-integration schema. Package B's script
# SUBSEQUENTLY landed in this same checkout (concurrent work, uncommitted at the time this comment
# was updated) while this package was still writing plot_results.py -- its ACTUAL
# `write_csv()` was read directly and confirms: `target` (per-target grouping) and `arm` (learned-
# arm label, present when more than one learned arm's rows share the file) were already this
# script's first-choice candidates and match exactly; the offline value and closed-loop delta
# columns are `offline_learned` (the learned model's own held-out AUC/RMSE point estimate) and
# `aligned_cl_delta` (B's own per-target ALIGNMENT map -- e.g. `satisfaction`/`not_interested` ->
# `mean_satisfaction`, `session_exit` -> `mean_session_utility`, everything else ->
# `reward_per_impression` -- resolved to the single closed-loop metric most relevant to that
# target, exactly the "one representative delta per target" this function needs), NEITHER of which
# was in this function's original speculative candidate list -- both are now prepended as the
# FIRST (confirmed-real) candidates, with the original guesses kept as fallbacks in case a
# differently-shaped gap_analysis.csv is ever pointed at this function. `offline_metric_type` (B's
# real column disambiguating AUC- from RMSE-valued targets) is read the same defensive way and, if
# found, appended to each group's x-axis label so mixed binary/continuous targets sharing one
# y-axis are not misread as directly comparable.


def plot_multiobjective_frontier(runs: list[RunData], outdir: Path,
                                 filename: str = "multiobjective_frontier.png") -> bool:
    """Phase 23 headline figure (V2 TDD §10 item 8, docs/design/P23-CONTRACTS.md §6 / plan Phase 23
    task 3b): scatter of x = engagement (summary.json metrics.reward_per_impression) vs. y = mean
    hidden satisfaction (welfare.mean_immediate_satisfaction), marker size encoding
    long_term.retention_7d (documented in a legend note -- matplotlib scatter's `s` is already an
    area-proportional-to-value encoding, the same convention plot_calibration's per-bin-count
    marker sizing established in Phase 22), one labeled point per run -- meant to be called with
    one RunData per arm from the Phase 23 nine-arm matrix (hand_tuned / semantic / learned(=
    w_balanced) / learned_survey_off / w_sat_100 / w_sat_70 / w_watch_70 / w_watch_100 /
    w_watch_100_noexit, contracts §4). Skipped for any run missing reward_per_impression or
    mean_immediate_satisfaction (same requirement as plot_engagement_vs_welfare); retention_7d is
    OPTIONAL per-run -- a run without a usable long_term block still plots, at a fixed neutral
    marker size, rather than being dropped (long_term is gated separately, P20 gates, from welfare,
    P14/15 gate -- a hand_tuned/semantic control arm could plausibly have one without the other).
    """
    usable = []
    for run in runs:
        if run.summary is None:
            continue
        engagement = run.summary.get("metrics", {}).get("reward_per_impression")
        welfare = run.summary.get("welfare")
        if engagement is None or not isinstance(welfare, dict):
            continue
        satisfaction = welfare.get("mean_immediate_satisfaction")
        if satisfaction is None:
            continue
        long_term = run.summary.get("long_term")
        retention = long_term.get("retention_7d") if isinstance(long_term, dict) else None
        if not (isinstance(retention, (int, float)) and not isinstance(retention, bool)):
            retention = None
        usable.append((run, engagement, satisfaction, retention))

    if not usable:
        warn(f"skipping {filename}: no run has both metrics.reward_per_impression and "
             f"welfare.mean_immediate_satisfaction (welfare requires realism.latent_reactions)")
        return False

    min_area, max_area, neutral_area = 80.0, 500.0, 160.0
    retentions = [r for _run, _e, _s, r in usable if r is not None]
    lo, hi = (min(retentions), max(retentions)) if retentions else (None, None)

    def area_for(retention):
        if retention is None or lo is None or hi is None or hi <= lo:
            return neutral_area
        return min_area + (retention - lo) / (hi - lo) * (max_area - min_area)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, engagement, satisfaction, retention in usable:
        ax.scatter(engagement, satisfaction, s=area_for(retention), zorder=3, color=run.color,
                  label=run.label, edgecolor="white", linewidth=0.6)
        ax.annotate(run.label, (engagement, satisfaction), textcoords="offset points",
                   xytext=(8, 4), fontsize=8)
    ax.text(0.02, 0.98, "marker area ∝ retention_7d  (fixed size when a run has none)",
           transform=ax.transAxes, va="top", fontsize=7, color="0.4")
    ax.margins(0.18)  # headroom so corner point annotations are not clipped
    finish_figure(fig, ax, outdir / filename,
                 "engagement  (reward per impression)",
                 "mean hidden satisfaction",
                 "Multi-Objective Frontier: Engagement vs. Hidden Satisfaction")
    return True


# First entry in each list is package B's CONFIRMED real scripts/phase23_gap_analysis.py column
# (read directly from its write_csv()); the rest are this function's original pre-coordination
# guesses, kept as defensive fallbacks (see section header comment above).
_P23_GAP_TARGET_CANDIDATES = ["target", "model_target", "metric_target", "metric"]
_P23_GAP_ARM_CANDIDATES = ["arm", "learned_arm", "model"]
_P23_GAP_OFFLINE_CANDIDATES = [
    "offline_learned", "offline_auc_or_rmse", "offline_value", "offline_metric_value",
    "offline_metric", "offline_auc", "offline_rmse",
]
_P23_GAP_DELTA_CANDIDATES = [
    "aligned_cl_delta", "closed_loop_delta", "closed_loop_delta_vs_hand_tuned", "closedloop_delta",
    "cl_delta", "delta",
]
_P23_GAP_METRIC_TYPE_CANDIDATES = ["offline_metric_type", "metric_type"]


def _first_present_column(columns, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def plot_offline_closedloop_gap(gap_csv, outdir: Path,
                                filename: str = "offline_closedloop_gap.png") -> bool:
    """Phase 23 offline-vs-closed-loop gap figure (docs/design/P23-CONTRACTS.md §6 / plan Phase 23
    task 4): grouped bar chart, one group per TARGET (or per target+arm when more than one learned
    arm's rows are present in `gap_csv`), two bars per group -- the target's offline held-out
    metric (AUC or RMSE; package B's `offline_learned` column) and its closed-loop delta vs.
    hand_tuned (package B's own per-target ALIGNMENT-map choice of the single most-relevant
    closed-loop metric, its `aligned_cl_delta` column, contracts §4). Reads `gap_csv`, package B's
    `gap_analysis.csv` (scripts/phase23_gap_analysis.py) -- its real column names were confirmed by
    reading that script's `write_csv()` directly once it landed in this same checkout (see this
    section's header comment for the exact mapping); column names are still discovered
    DEFENSIVELY by candidate search (the confirmed names tried first, this function's original
    pre-coordination guesses kept as fallbacks) rather than assumed as a single frozen header the
    way e.g. plot_offline_auc reads Phase 22's training_eval.csv, since gap_analysis.csv's header
    is not a contracts-frozen literal. When an `offline_metric_type` column is present (B's real
    AUC-vs-RMSE disambiguator), it is appended to each group's x-axis label so targets of different
    metric types sharing one y-axis are not misread as directly comparable. Skipped (warn) when the
    file is absent/unreadable, or a required target/offline-value/delta-value column cannot be
    found under any candidate name.
    """
    df = _read_csv(Path(gap_csv))
    if df is None:
        warn(f"skipping {filename}: {gap_csv} missing or unreadable")
        return False

    target_col = _first_present_column(df.columns, _P23_GAP_TARGET_CANDIDATES)
    offline_col = _first_present_column(df.columns, _P23_GAP_OFFLINE_CANDIDATES)
    delta_col = _first_present_column(df.columns, _P23_GAP_DELTA_CANDIDATES)
    if target_col is None or offline_col is None or delta_col is None:
        warn(f"skipping {filename}: {gap_csv} -- could not find a target/offline-value/"
             f"closed-loop-delta column under any candidate name (columns found: "
             f"{list(df.columns)}; candidates tried: target={_P23_GAP_TARGET_CANDIDATES}, "
             f"offline={_P23_GAP_OFFLINE_CANDIDATES}, delta={_P23_GAP_DELTA_CANDIDATES})")
        return False

    arm_col = _first_present_column(df.columns, _P23_GAP_ARM_CANDIDATES)
    metric_type_col = _first_present_column(df.columns, _P23_GAP_METRIC_TYPE_CANDIDATES)
    df = df.dropna(subset=[target_col, offline_col, delta_col])
    if df.empty:
        warn(f"skipping {filename}: no usable (non-NaN) row in {gap_csv}")
        return False

    def _label(row) -> str:
        label = str(row[target_col])
        if arm_col is not None and df[arm_col].nunique() > 1:
            label += f"\n({row[arm_col]})"
        if metric_type_col is not None and row[metric_type_col]:
            label += f" [{row[metric_type_col]}]"
        return label

    group_labels = [_label(row) for _idx, row in df.iterrows()]
    offline_vals = df[offline_col].tolist()
    delta_vals = df[delta_col].tolist()

    x = list(range(len(group_labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(FIGSIZE[0], 1.2 * len(group_labels)), FIGSIZE[1]))
    ax.bar([i - width / 2 for i in x], offline_vals, width, color=COLOR_CYCLE[0],
          label=f"offline ({offline_col})", zorder=3)
    ax.bar([i + width / 2 for i in x], delta_vals, width, color=COLOR_CYCLE[1],
          label=f"closed-loop delta vs hand_tuned ({delta_col})", zorder=3)
    ax.axhline(0.0, color="0.5", linestyle="-", linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right", fontsize=8)
    finish_figure(fig, ax, outdir / filename, "target", "value",
                 "Offline Held-Out Metric vs. Closed-Loop Delta (vs. hand_tuned)")
    return True


# --- Phase 11 benchmark plots (retrieval_metrics.csv / load_metrics.csv) ----------------------
#
# Each returns the number of PNGs written (0 => nothing usable, warn-skipped), summed in main().
# Retrieval plots facet by grid axes so "lines per M" stays unambiguous; latency plots exclude
# distance-counting rows (distance_comps_per_query >= 0), whose latency is instrumentation-inflated.


def plot_recall_vs_efsearch(retr: Optional[pd.DataFrame], outdir: Path) -> int:
    required = {"k", "ef_search", "m", "recall_at_k", "vector_count", "dimensions",
                "ef_construction"}
    if retr is None or not required <= set(retr.columns):
        warn("skipping recall_vs_efsearch: no retrieval_metrics.csv with the required columns")
        return 0
    k10 = retr[retr["k"] == 10]
    if k10.empty:
        warn("skipping recall_vs_efsearch: no k=10 rows in retrieval_metrics.csv")
        return 0

    written = 0
    # One figure per (vector_count, dimensions, data_distribution, ef_construction) so each line is
    # a clean M-sweep at a fixed efConstruction.
    for (vc, dim, dist, efc), g in k10.groupby(
        ["vector_count", "dimensions", "data_distribution", "ef_construction"]
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for i, (m, gm) in enumerate(sorted(g.groupby("m"))):
            gm = gm.sort_values("ef_search")
            ax.plot(gm["ef_search"], gm["recall_at_k"], marker="o",
                    color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=f"M={m}")
        ax.set_xscale("log", base=2)
        ax.axhline(0.90, color="gray", linestyle=":", linewidth=1)  # TDD 27 Recall@10 target.
        out = outdir / f"recall_vs_efsearch_vc{vc}_d{dim}_{dist}_efc{efc}.png"
        finish_figure(fig, ax, out, "efSearch (log2)", "Recall@10",
                      f"Recall@10 vs efSearch  (N={vc}, d={dim}, {dist}, efC={efc})")
        written += 1
    return written


def plot_latency_vs_efsearch(retr: Optional[pd.DataFrame], outdir: Path) -> int:
    """Query p95 latency vs efSearch, one line per M, one figure per grid cell (§26 graph).

    Mirrors plot_recall_vs_efsearch's faceting so each line is a clean M-sweep. Only clean-latency
    rows (distance_comps_per_query < 0) are used: distance-counting passes inflate the timer.
    """
    required = {"k", "ef_search", "m", "query_p95_ms", "vector_count", "dimensions",
                "ef_construction"}
    if retr is None or not required <= set(retr.columns):
        warn("skipping latency_vs_efsearch: no retrieval_metrics.csv with the required columns")
        return 0
    df = retr[retr["k"] == 10]
    if "distance_comps_per_query" in df.columns:
        df = df[df["distance_comps_per_query"] < 0]
    if df.empty:
        warn("skipping latency_vs_efsearch: no clean-latency k=10 rows")
        return 0

    written = 0
    for (vc, dim, dist, efc), g in df.groupby(
        ["vector_count", "dimensions", "data_distribution", "ef_construction"]
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for i, (m, gm) in enumerate(sorted(g.groupby("m"))):
            gm = gm.sort_values("ef_search")
            ax.plot(gm["ef_search"], gm["query_p95_ms"], marker="o",
                    color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=f"M={m}")
        ax.set_xscale("log", base=2)
        out = outdir / f"latency_vs_efsearch_vc{vc}_d{dim}_{dist}_efc{efc}.png"
        finish_figure(fig, ax, out, "efSearch (log2)", "query p95 latency (ms)",
                      f"Query p95 Latency vs efSearch  (N={vc}, d={dim}, {dist}, efC={efc})")
        written += 1
    return written


def plot_recall_vs_latency(retr: Optional[pd.DataFrame], outdir: Path) -> int:
    required = {"k", "recall_at_k", "query_p95_ms", "m", "ef_search", "vector_count", "dimensions"}
    if retr is None or not required <= set(retr.columns):
        warn("skipping recall_vs_latency: no retrieval_metrics.csv with the required columns")
        return 0
    df = retr[retr["k"] == 10]
    # Latency hygiene: drop distance-counting rows (flagged distance_comps_per_query >= 0) whose
    # p95 includes atomic-increment overhead.
    if "distance_comps_per_query" in df.columns:
        df = df[df["distance_comps_per_query"] < 0]
    if df.empty:
        warn("skipping recall_vs_latency: no clean-latency k=10 rows")
        return 0

    written = 0
    for (vc, dim, dist), g in df.groupby(["vector_count", "dimensions", "data_distribution"]):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for i, (m, gm) in enumerate(sorted(g.groupby("m"))):
            ax.scatter(gm["query_p95_ms"], gm["recall_at_k"],
                       color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=f"M={m}", zorder=3)
            for _, row in gm.iterrows():
                ax.annotate(f"ef{int(row['ef_search'])}",
                            (row["query_p95_ms"], row["recall_at_k"]),
                            textcoords="offset points", xytext=(3, 3), fontsize=6)
        ax.axhline(0.90, color="gray", linestyle=":", linewidth=1)
        out = outdir / f"recall_vs_latency_vc{vc}_d{dim}_{dist}.png"
        finish_figure(fig, ax, out, "query p95 latency (ms)", "Recall@10",
                      f"Recall@10 vs p95 latency  (N={vc}, d={dim}, {dist})")
        written += 1
    return written


def plot_throughput_vs_threads(load: Optional[pd.DataFrame], outdir: Path,
                               filename: str = "throughput_vs_threads.png") -> int:
    required = {"threads", "rps", "corpus_reels", "dimensions"}
    if load is None or not required <= set(load.columns):
        warn(f"skipping {filename}: no load_metrics.csv with the required columns")
        return 0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for i, ((corpus, dim), g) in enumerate(sorted(load.groupby(["corpus_reels", "dimensions"]))):
        g = g.sort_values("threads")
        ax.plot(g["threads"], g["rps"], marker="o",
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=f"N={corpus}, d={dim}")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn(f"skipping {filename}: no usable load_metrics.csv rows")
        return 0
    finish_figure(fig, ax, outdir / filename, "client threads",
                  "recommendations / second", "Throughput vs Concurrent Clients")
    return 1


def plot_p99_vs_corpus(load: Optional[pd.DataFrame], outdir: Path) -> int:
    required = {"corpus_reels", "threads", "e2e_p99_ms", "retrieval_p99_ms"}
    if load is None or not required <= set(load.columns):
        warn("skipping p99_vs_corpus: no load_metrics.csv with the required columns")
        return 0
    max_threads = load["threads"].max()
    at_max = load[load["threads"] == max_threads]
    has_dim = "dimensions" in at_max.columns
    groups = at_max.groupby("dimensions") if has_dim else [(None, at_max)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for i, (dim, g) in enumerate(groups):
        g = g.sort_values("corpus_reels")
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        suffix = f" (d={dim})" if dim is not None else ""
        ax.plot(g["corpus_reels"], g["e2e_p99_ms"], marker="o", color=color,
                linestyle="-", label=f"end-to-end p99{suffix}")
        ax.plot(g["corpus_reels"], g["retrieval_p99_ms"], marker="s", color=color,
                linestyle="--", label=f"retrieval p99{suffix}")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn("skipping p99_vs_corpus: no usable load_metrics.csv rows")
        return 0
    ax.set_xscale("log")
    finish_figure(fig, ax, outdir / "p99_vs_corpus.png", "corpus size (reels)",
                  "p99 latency (ms)",
                  f"Tail Latency vs Corpus Size  (threads={int(max_threads)})")
    return 1


# --- Canonical published figure set (--canonical) ---------------------------------------------
#
# generate_canonical() renders EXACTLY the eleven §26 recommended graphs into --out with the exact
# filenames documented in results/published/figures/README.md. The three retrieval graphs below
# combine the clustered and random distributions at one fixed grid cell (unlike the faceted
# plot_recall_vs_efsearch / plot_latency_vs_efsearch, which emit one file per cell); the remaining
# eight reuse the general plotting functions with forced output filenames.


def _canonical_cell(retr: pd.DataFrame, m: Optional[int]) -> pd.DataFrame:
    """Subset a retrieval frame to the canonical grid cell, clean-latency rows, k=10."""
    df = retr[(retr["k"] == 10)
              & (retr["vector_count"] == CANONICAL_CELL["vector_count"])
              & (retr["dimensions"] == CANONICAL_CELL["dimensions"])
              & (retr["ef_construction"] == CANONICAL_CELL["ef_construction"])]
    # Drop distance-counting duplicate rows (distance_comps_per_query >= 0): their timer is inflated
    # and recall is identical to the clean pass, so this also de-duplicates each (dist, ef) point.
    if "distance_comps_per_query" in df.columns:
        df = df[df["distance_comps_per_query"] < 0]
    if m is not None:
        df = df[df["m"] == m]
    return df


def _distributions_in_order(df: pd.DataFrame) -> list[str]:
    present = list(df["data_distribution"].unique())
    return [d for d in DIST_ORDER if d in present] + [d for d in present if d not in DIST_ORDER]


def plot_recall_vs_efsearch_cell(retr: Optional[pd.DataFrame], outpath: Path,
                                 m: int = CANONICAL_M) -> bool:
    """Recall@10 vs efSearch at the canonical cell, one line per data distribution (fixed M)."""
    required = {"k", "ef_search", "m", "recall_at_k", "vector_count", "dimensions",
                "ef_construction", "data_distribution"}
    if retr is None or not required <= set(retr.columns):
        warn("skipping recall_vs_efsearch.png: retrieval frame missing required columns")
        return False
    df = _canonical_cell(retr, m)
    if df.empty:
        warn("skipping recall_vs_efsearch.png: no rows at the canonical cell")
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for dist in _distributions_in_order(df):
        g = df[df["data_distribution"] == dist].sort_values("ef_search")
        ax.plot(g["ef_search"], g["recall_at_k"], marker="o",
                color=DIST_COLOR.get(dist, COLOR_CYCLE[2]), label=f"{dist} (M={m})")
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1, label="Recall@10 target 0.90")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    c = CANONICAL_CELL
    finish_figure(fig, ax, outpath, "efSearch (log2)", "Recall@10",
                  f"Recall@10 vs efSearch  (N={c['vector_count']}, d={c['dimensions']}, "
                  f"efC={c['ef_construction']})")
    return True


def plot_latency_vs_efsearch_cell(retr: Optional[pd.DataFrame], outpath: Path,
                                  m: int = CANONICAL_M) -> bool:
    """Query p95 latency vs efSearch at the canonical cell, one line per data distribution."""
    required = {"k", "ef_search", "m", "query_p95_ms", "vector_count", "dimensions",
                "ef_construction", "data_distribution"}
    if retr is None or not required <= set(retr.columns):
        warn("skipping latency_vs_efsearch.png: retrieval frame missing required columns")
        return False
    df = _canonical_cell(retr, m)
    if df.empty:
        warn("skipping latency_vs_efsearch.png: no rows at the canonical cell")
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for dist in _distributions_in_order(df):
        g = df[df["data_distribution"] == dist].sort_values("ef_search")
        ax.plot(g["ef_search"], g["query_p95_ms"], marker="o",
                color=DIST_COLOR.get(dist, COLOR_CYCLE[2]), label=f"{dist} (M={m})")
    ax.set_xscale("log", base=2)
    c = CANONICAL_CELL
    finish_figure(fig, ax, outpath, "efSearch (log2)", "query p95 latency (ms)",
                  f"Query p95 Latency vs efSearch  (N={c['vector_count']}, d={c['dimensions']}, "
                  f"efC={c['ef_construction']})")
    return True


def plot_recall_vs_latency_cell(retr: Optional[pd.DataFrame], outpath: Path) -> bool:
    """Recall@10 vs query p95 latency Pareto at the canonical cell; color=distribution, marker=M."""
    required = {"k", "ef_search", "m", "recall_at_k", "query_p95_ms", "vector_count",
                "dimensions", "ef_construction", "data_distribution"}
    if retr is None or not required <= set(retr.columns):
        warn("skipping recall_vs_latency.png: retrieval frame missing required columns")
        return False
    df = _canonical_cell(retr, m=None)
    if df.empty:
        warn("skipping recall_vs_latency.png: no rows at the canonical cell")
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for dist in _distributions_in_order(df):
        for m, gm in sorted(df[df["data_distribution"] == dist].groupby("m")):
            gm = gm.sort_values("query_p95_ms")
            ax.plot(gm["query_p95_ms"], gm["recall_at_k"], linestyle="-", linewidth=0.8,
                    marker=MARKER_BY_M.get(m, "o"), color=DIST_COLOR.get(dist, COLOR_CYCLE[2]),
                    label=f"{dist} M={m}", zorder=3)
            for _, row in gm.iterrows():
                ax.annotate(f"ef{int(row['ef_search'])}",
                            (row["query_p95_ms"], row["recall_at_k"]),
                            textcoords="offset points", xytext=(3, 3), fontsize=6)
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.02)
    c = CANONICAL_CELL
    finish_figure(fig, ax, outpath, "query p95 latency (ms)", "Recall@10",
                  f"Recall@10 vs p95 Latency  (N={c['vector_count']}, d={c['dimensions']}, "
                  f"efC={c['ef_construction']})")
    return True


def _retrieval_lat_row(latency: Optional[pd.DataFrame], stage: str) -> Optional[pd.Series]:
    if latency is None or "stage" not in latency.columns:
        return None
    rows = latency[latency["stage"] == stage]
    return rows.iloc[0] if not rows.empty else None


def plot_hnsw_vs_exact(exact: dict, hnsw: dict, outpath: Path) -> bool:
    """Two-panel HNSW-vs-exact: quality parity (left) and retrieval latency gap (right, log-y).

    Each arm dict is {label, summary, latency}. Quality reads summary.json; latency reads the
    retrieval-stage row of latency_metrics.csv (p50/p99 percentiles, robust to the exact arm's
    machine-sleep-contaminated mean/max).
    """
    def q(arm, section, key):
        s = arm.get("summary") or {}
        return s.get(section, {}).get(key)

    quality = [
        ("mean true\naffinity", "metrics", "mean_true_affinity"),
        ("reward /\nimpression", "metrics", "reward_per_impression"),
        ("oracle mean\nregret", "oracle", "mean_regret"),
    ]
    exact_q = [q(exact, sec, key) for _, sec, key in quality]
    hnsw_q = [q(hnsw, sec, key) for _, sec, key in quality]
    lat_exact = _retrieval_lat_row(exact.get("latency"), "retrieval")
    lat_hnsw = _retrieval_lat_row(hnsw.get("latency"), "retrieval")
    if None in exact_q or None in hnsw_q or lat_exact is None or lat_hnsw is None:
        warn("skipping hnsw_vs_exact.png: missing summary.json metrics or latency_metrics.csv rows")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    labels = [q[0] for q in quality]
    x = list(range(len(labels)))
    w = 0.38
    b1 = ax1.bar([i - w / 2 for i in x], exact_q, w, color=COLOR_CYCLE[0], label=exact["label"])
    b2 = ax1.bar([i + w / 2 for i in x], hnsw_q, w, color=COLOR_CYCLE[1], label=hnsw["label"])
    ax1.bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
    ax1.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("value (reward / affinity / regret units)")
    ax1.set_title("Recommendation quality (parity)")
    ax1.grid(alpha=GRID_ALPHA, axis="y")
    ax1.legend(fontsize=8)

    stages = ["retrieval p50", "retrieval p99"]
    xe = list(range(len(stages)))
    exact_lat = [float(lat_exact["p50_ms"]), float(lat_exact["p99_ms"])]
    hnsw_lat = [float(lat_hnsw["p50_ms"]), float(lat_hnsw["p99_ms"])]
    l1 = ax2.bar([i - w / 2 for i in xe], exact_lat, w, color=COLOR_CYCLE[0], label=exact["label"])
    l2 = ax2.bar([i + w / 2 for i in xe], hnsw_lat, w, color=COLOR_CYCLE[1], label=hnsw["label"])
    ax2.bar_label(l1, fmt="%.2f", fontsize=7, padding=2)
    ax2.bar_label(l2, fmt="%.3f", fontsize=7, padding=2)
    ax2.set_yscale("log")
    ax2.set_xticks(xe)
    ax2.set_xticklabels(stages, fontsize=8)
    ax2.set_ylabel("retrieval latency (ms, log scale)")
    speedup = exact_lat[1] / hnsw_lat[1] if hnsw_lat[1] else float("nan")
    ax2.set_title(f"Retrieval latency  ({speedup:.0f}x lower p99)")
    ax2.grid(alpha=GRID_ALPHA, axis="y")
    ax2.legend(fontsize=8)

    fig.suptitle("HNSW vs Exact Vector Search")
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    return True


def plot_diversity_vs_reward(arms: list[dict], outpath: Path) -> bool:
    """Scatter reward/impression vs intra-list similarity, one labeled point per re-ranking arm."""
    usable = [a for a in arms if a.get("reward") is not None and a.get("ils") is not None]
    if not usable:
        warn("skipping diversity_vs_reward.png: no arm has reward + intra-list similarity")
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, arm in enumerate(usable):
        ax.scatter(arm["ils"], arm["reward"], s=110, zorder=3,
                   color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=arm["label"])
        note = arm["label"]
        if arm.get("topics") is not None:
            note += f"\n({arm['topics']:.2f} topics/feed)"
        ax.annotate(note, (arm["ils"], arm["reward"]), textcoords="offset points",
                    xytext=(8, 4), fontsize=7)
    ax.margins(0.18)  # headroom so corner point annotations are not clipped
    finish_figure(fig, ax, outpath,
                  "intra-list similarity  (lower = more diverse)",
                  "reward per impression",
                  "Diversity vs Reward Trade-off")
    return True


def plot_cold_start(arms: list[dict], outpath: Path, smooth: int = 5) -> bool:
    """Injected new-user mean reward vs impressions-since-injection, one line per exploration arm.

    Each series is a per-impression-index mean over the injected-user cohort, so it is inherently
    noisy; a centered rolling mean (window `smooth`, min_periods=1) is drawn to expose the warming
    trend, with the raw series kept as a faint underlay. Pass smooth<=1 to draw the raw series only.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for i, arm in enumerate(arms):
        df = arm.get("df")
        if df is None or "impression_index" not in df.columns or "mean_reward" not in df.columns:
            continue
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        x = df["impression_index"]
        if smooth > 1:
            ax.plot(x, df["mean_reward"], color=color, alpha=0.18, linewidth=0.8)
            y = df["mean_reward"].rolling(smooth, center=True, min_periods=1).mean()
        else:
            y = df["mean_reward"]
        ax.plot(x, y, color=color, label=arm["label"])
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        warn("skipping cold_start.png: no arm has usable new_user_curve.csv data")
        return False
    title = "Cold-Start Learning Curve"
    if smooth > 1:
        title += f"  ({smooth}-impression rolling mean)"
    finish_figure(fig, ax, outpath,
                  "impressions since user injection",
                  "mean reward per impression (injected users)",
                  title)
    return True


def _glob_one(root: Path, pattern: str, what: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    if not matches:
        warn(f"canonical: no directory matching {root}/{pattern} ({what})")
        return None
    return matches[0]


def _load_arm(directory: Optional[Path]) -> Optional[dict]:
    if directory is None or not directory.is_dir():
        return None
    return {"summary": _read_json(directory / "summary.json"),
            "latency": _read_csv(directory / "latency_metrics.csv")}


def _load_cold_start_arm(directory: Path, label: str) -> Optional[dict]:
    if not directory.is_dir():
        warn(f"canonical: cold-start dir not found: {directory}")
        return None
    df = _read_csv(directory / "new_user_curve.csv")
    config = _read_json(directory / "config.json")
    if config is not None:
        eps = config.get("exploration", {}).get("epsilon")
        if eps is not None:
            label = f"ε={eps:g}"
    return {"label": label, "df": df}


def _load_diversity_arm(directory: Path, label: str) -> Optional[dict]:
    if not directory.is_dir():
        warn(f"canonical: diversity dir not found: {directory}")
        return None
    s = _read_json(directory / "summary.json")
    if s is None:
        return None
    div = s.get("diversity", {})
    return {"label": label,
            "reward": s.get("metrics", {}).get("reward_per_impression"),
            "ils": div.get("mean_intra_list_similarity"),
            "topics": div.get("mean_unique_topics")}


def generate_canonical(published_root: Path, outdir: Path, only: Optional[set]) -> int:
    """Render the eleven canonical §26 figures into outdir. Returns the number written."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Retrieval / load sweeps (Phase 11).
    retr = _combined_retrieval([load_benchmark(published_root / "phase11" / "retrieval", "retr")])
    load = _combined_load([load_benchmark(published_root / "phase11" / "load", "load")])

    # Phase 7 online-learning arms (learning vs frozen).
    p7 = published_root / "phase7"
    p7_runs = []
    for pat, label, ci in [("hnsw_ranker-seed42-*", "learning", 0),
                           ("frozen-hnsw_ranker-seed42-*", "frozen", 1)]:
        d = _glob_one(p7, pat, f"phase7 {label}")
        if d is not None:
            p7_runs.append(load_run(d, label, COLOR_CYCLE[ci]))

    # Phase 5 exact vs HNSW arms.
    p5 = published_root / "phase5"
    exact_arm = _load_arm(_glob_one(p5, "exact_vector-seed42-*", "phase5 exact"))
    hnsw_arm = _load_arm(_glob_one(p5, "hnsw-seed42-*", "phase5 hnsw"))
    if exact_arm is not None:
        exact_arm["label"] = "exact"
    if hnsw_arm is not None:
        hnsw_arm["label"] = "HNSW"

    # Phase 9 re-ranking arms (aggregate reward vs diversity).
    p9 = published_root / "phase9"
    p9_arms = [a for a in (
        _load_diversity_arm(p9 / "hnsw_ranker", "ranker only"),
        _load_diversity_arm(p9 / "constraints", "+ constraints"),
        _load_diversity_arm(p9 / "constraints-mmr", "+ constraints + MMR"),
        _load_diversity_arm(p9 / "complete-system", "complete system"),
    ) if a is not None]

    # Phase 8 cold-start exploration arms.
    p8 = published_root / "phase8"
    p8_arms = [a for a in (
        _load_cold_start_arm(p8 / "eps000", "ε=0"),
        _load_cold_start_arm(p8 / "eps002", "ε=0.02"),
        _load_cold_start_arm(p8 / "eps005", "ε=0.05"),
        _load_cold_start_arm(p8 / "eps010", "ε=0.1"),
    ) if a is not None]

    # Phase 10 drift arms.
    p10 = published_root / "phase10"
    p10_runs = []
    for name, label, ci in [("session_aware", "session-aware", 0),
                            ("long_term_only", "long-term only", 1),
                            ("frozen", "frozen", 2)]:
        d = p10 / name
        if d.is_dir():
            p10_runs.append(load_run(d, label, COLOR_CYCLE[ci]))
        else:
            warn(f"canonical: phase10 dir not found: {d}")

    # name -> thunk producing exactly one PNG. Order matches the figures README.
    figures = {
        "recall_vs_efsearch":
            lambda: plot_recall_vs_efsearch_cell(retr, outdir / "recall_vs_efsearch.png"),
        "latency_vs_efsearch":
            lambda: plot_latency_vs_efsearch_cell(retr, outdir / "latency_vs_efsearch.png"),
        "recall_vs_latency":
            lambda: plot_recall_vs_latency_cell(retr, outdir / "recall_vs_latency.png"),
        "reward_vs_interactions":
            lambda: plot_reward_curve(p7_runs, outdir, "reward_vs_interactions.png"),
        "cumulative_regret":
            lambda: plot_cumulative_regret(p7_runs, outdir),
        "hnsw_vs_exact":
            lambda: (exact_arm is not None and hnsw_arm is not None
                     and plot_hnsw_vs_exact(exact_arm, hnsw_arm, outdir / "hnsw_vs_exact.png"))
                    or _skip("hnsw_vs_exact.png: phase5 arm dirs missing"),
        "diversity_vs_reward":
            lambda: plot_diversity_vs_reward(p9_arms, outdir / "diversity_vs_reward.png"),
        "cold_start":
            lambda: plot_cold_start(p8_arms, outdir / "cold_start.png"),
        "drift_recovery":
            lambda: plot_drift_recovery(p10_runs, outdir, None),
        "throughput_vs_clients":
            lambda: bool(plot_throughput_vs_threads(load, outdir, "throughput_vs_clients.png")),
        "p99_vs_corpus":
            lambda: bool(plot_p99_vs_corpus(load, outdir)),
    }

    if only:
        unknown = only - set(figures)
        if unknown:
            warn(f"canonical: unknown --only figure(s): {', '.join(sorted(unknown))}")
        selected = [(n, t) for n, t in figures.items() if n in only]
    else:
        selected = list(figures.items())

    written = 0
    for name, thunk in selected:
        if thunk():
            written += 1
    return written


def _skip(message: str) -> bool:
    warn(f"skipping {message}")
    return False


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
        "result_dirs", nargs="*", metavar="result-dir",
        help="one or more experiment result directories (omit with --canonical)",
    )
    parser.add_argument(
        "--labels", default=None,
        help="comma-separated run labels, one per result-dir (default: directory basenames)",
    )
    parser.add_argument(
        "--out", default="plots",
        help="output directory for plots (default: plots; with --canonical, "
             "results/published/figures under the repo root)",
    )
    parser.add_argument(
        "--drift-at", type=int, default=None, dest="drift_at",
        help="interactions_per_user marking drift onset, used when no run's "
             "summary.json has adaptation.first_drift_interaction",
    )
    parser.add_argument(
        "--canonical", action="store_true",
        help="regenerate the eleven canonical §26 figures from the published tree into --out "
             "(ignores result-dir positional args)",
    )
    parser.add_argument(
        "--published-root", default=None, dest="published_root",
        help="root of the published results tree for --canonical "
             "(default: <repo>/results/published)",
    )
    parser.add_argument(
        "--only", default=None,
        help="with --canonical, comma-separated figure name(s) to (re)generate, e.g. "
             "recall_vs_efsearch,cold_start (default: all eleven)",
    )
    parser.add_argument(
        "--phase20", nargs=2, default=None, metavar=("ENGAGEMENT_ON_DIR", "PROXY_ON_DIR"),
        help="Phase 20 preference-divergence histogram (V2 TDD §4.15-4.17, Tier-4 acceptance item "
             "1): engagement-on and proxy-on result directories, each must contain "
             "hidden_preference_final.csv (contracts §5). Independent of the positional result-dir "
             "argument, which (when given) also drives the Phase 20 retention_by_day.png / "
             "trust_trajectory.png plots from longterm_metrics.csv -- combine both in one call, "
             "e.g. `plot_results.py <4 arm dirs> --phase20 <engagement-on dir> <proxy-on dir>`.",
    )
    parser.add_argument(
        "--phase22", default=None, metavar="EVAL_CSV",
        help="Phase 22 offline AUC bar chart (V2 TDD §4.19-4.20, docs/design/P22-CONTRACTS.md "
             "§6): a single split's training_eval.csv (contracts §5 frozen header) for "
             "offline_auc.png (grouped bars, learned + 3 baselines, binary targets only). "
             "Independent of the positional result-dir argument and of --phase22-calibration; "
             "call once per split (e.g. results/phase22/models-temporal/training_eval.csv).",
    )
    parser.add_argument(
        "--phase22-calibration", default=None, dest="phase22_calibration", metavar="DIR",
        help="Phase 22 reliability diagrams: a directory containing one or more "
             "calibration-<target>.csv files (contracts §5 frozen header) for calibration_"
             "<target>.png, one per file found by glob (e.g. "
             "results/phase22/models-temporal, or scripts/phase22_report.py's own "
             "<out>/calibration/<slug>/). Independent of --phase22; omit to skip calibration "
             "plots.",
    )
    parser.add_argument(
        "--phase23", default=None, metavar="GAP_CSV",
        help="Phase 23 offline-vs-closed-loop gap chart (docs/design/P23-CONTRACTS.md §6): "
             "package B's gap_analysis.csv (scripts/phase23_gap_analysis.py) for "
             "offline_closedloop_gap.png (grouped bars, one group per target, its offline_learned "
             "metric vs. its aligned_cl_delta vs. hand_tuned). Column names are discovered "
             "defensively (candidate search, package B's confirmed real names tried first) -- see "
             "plot_offline_closedloop_gap's docstring. Independent of the positional result-dir "
             "argument, which (when given, one dir per Phase 23 arm) also drives "
             "multiobjective_frontier.png. Omit to skip this one chart.",
    )
    return parser.parse_args(argv)


REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.canonical:
        published_root = (Path(args.published_root) if args.published_root
                          else REPO_ROOT / "results" / "published")
        outdir = Path(args.out) if args.out != "plots" else REPO_ROOT / "results" / "published" / "figures"
        only = {s.strip() for s in args.only.split(",")} if args.only else None
        written = generate_canonical(published_root, outdir, only)
        if written:
            print(f"plot_results: wrote {written} canonical figure(s) to {outdir}")
        return 0 if written > 0 else 1

    if not args.result_dirs:
        warn("no result directories given (use --canonical to regenerate the published figure set)")
        return 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    dirs = [Path(d) for d in args.result_dirs]
    labels = derive_labels(args.labels, dirs)
    colors = [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(len(dirs))]

    runs = [load_run(d, label, color) for d, label, color in zip(dirs, labels, colors)]

    written = 0
    # Simulation curves (learning_curve.csv / regret_curve.csv). Each warn-skips when absent.
    written += plot_reward_curve(runs, outdir)
    written += plot_alignment_curve(runs, outdir)
    written += plot_cumulative_regret(runs, outdir)
    written += plot_drift_recovery(runs, outdir, args.drift_at)
    written += plot_engagement_vs_welfare(runs, outdir)
    written += plot_freshness_cost_frontier(runs, outdir)

    # Phase 20 additions (V2 TDD §4.15-4.17, Tier-4 acceptance item 1). Independent of the RunData
    # list above -- see plot_preference_divergence_hist / plot_retention_by_day /
    # plot_trust_trajectory's docstrings for why each has its own tiny loader (Phase20Run) instead
    # of reusing RunData (this file's existing functions/dataclass are left untouched, per this
    # package's brief). retention_by_day.png / trust_trajectory.png warn-skip cleanly (the
    # established pattern) whenever no positional dir has a longterm_metrics.csv -- e.g. a plain
    # Phase 7/9/... call with no Phase 20 data still runs cleanly, just without those two PNGs.
    if args.phase20:
        engagement_on_dir, proxy_on_dir = Path(args.phase20[0]), Path(args.phase20[1])
        written += plot_preference_divergence_hist(engagement_on_dir, proxy_on_dir, outdir)
    phase20_runs = [_load_phase20_run(d, label, color) for d, label, color in zip(dirs, labels, colors)]
    written += plot_retention_by_day(phase20_runs, outdir)
    written += plot_trust_trajectory(phase20_runs, outdir)

    # Phase 21 additions (V2 TDD §4.18, docs/design/P21-CONTRACTS.md §6). creator_hhi_by_day.png /
    # archetype_share_by_day_<label>.png need ecosystem_metrics.csv (Phase21Run, a new loader
    # mirroring Phase20Run above); entropy_by_day.png reuses the EXISTING phase20_runs list
    # (Phase20Run already reads longterm_metrics.csv in full); retention_welfare_frontier.png
    # reuses the EXISTING runs list (RunData already reads summary.json). All four warn-skip
    # cleanly when a plain pre-Phase-21 call has none of this data, exactly like the Phase 20
    # additions above.
    phase21_runs = [_load_phase21_run(d, label, color) for d, label, color in zip(dirs, labels, colors)]
    written += plot_creator_hhi_by_day(phase21_runs, outdir)
    written += plot_archetype_share_by_day(phase21_runs, outdir)
    written += plot_entropy_by_day(phase20_runs, outdir)
    written += plot_retention_welfare_frontier(runs, outdir)

    # Phase 22 additions (V2 TDD §4.19-4.20/§4.22, docs/design/P22-CONTRACTS.md §6). Both flags
    # are independent of the positional result-dir arguments and of each other (see their own
    # parse_args help text above) -- a plain pre-Phase-22 call with neither flag set is unaffected.
    if args.phase22:
        written += plot_offline_auc(args.phase22, outdir)
    if args.phase22_calibration:
        calibration_csvs = sorted(Path(args.phase22_calibration).glob("calibration-*.csv"))
        written += plot_calibration(calibration_csvs, outdir)

    # Phase 23 additions (learned multi-objective ranking in the loop, docs/design/
    # P23-CONTRACTS.md §6). plot_multiobjective_frontier reuses the EXISTING `runs` list (RunData
    # already loaded summary.json content -- same shape as plot_engagement_vs_welfare/
    # plot_retention_welfare_frontier above, called but not modified, per this package's
    # append-only brief); meant to be called with one positional result-dir per arm from the Phase
    # 23 nine-arm matrix (contracts §4). plot_offline_closedloop_gap is independent of the
    # positional result-dir arguments (its own CSV input, package B's gap_analysis.csv, contracts
    # §4) -- gated behind --phase23, so a plain pre-Phase-23 call is unaffected.
    written += plot_multiobjective_frontier(runs, outdir)
    if args.phase23:
        written += plot_offline_closedloop_gap(args.phase23, outdir)

    # Phase 11 benchmark plots (retrieval_metrics.csv / load_metrics.csv). Same dirs; a dir may hold
    # simulation CSVs, benchmark CSVs, or (a parent dir) several benchmark subtrees. Each warn-skips.
    benchruns = [load_benchmark(d, label) for d, label in zip(dirs, labels)]
    retr = _combined_retrieval(benchruns)
    load = _combined_load(benchruns)
    written += plot_recall_vs_efsearch(retr, outdir)
    written += plot_latency_vs_efsearch(retr, outdir)
    written += plot_recall_vs_latency(retr, outdir)
    written += plot_throughput_vs_threads(load, outdir)
    written += plot_p99_vs_corpus(load, outdir)

    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
