#!/usr/bin/env python3
"""phase23_gap_analysis.py — Phase 23 offline-vs-closed-loop gap analysis (V2 TDD §4.21 + Tier-5
acceptance item 5 [plan Phase 23 task 4]; docs/design/P23-CONTRACTS.md §4 [this package's own spec]).
Package B.

The Phase 23 closed-loop matrix (scripts/run_phase23_experiment.sh) serves the learned §4.21
multi-objective value function IN the loop on worlds identical to the hand-tuned baseline. This
script quantifies the GAP between (a) how well each learned arm's models predict held-out OBSERVABLE
outcomes OFFLINE — the P22 held-out AUC/RMSE, re-evaluated post-hoc on the arm's OWN in-run
training_log (scripts/run_phase23_experiment.sh prints the `train_models ... --split temporal`
re-eval per learned arm) — and (b) what serving that learned value ACTUALLY did closed-loop, as
deltas of each arm's engagement/welfare/session-health/retention outcomes against the baseline arm.

The gap is the point of acceptance 5: a model can be offline-predictive yet drive a neutral or
NEGATIVE closed-loop outcome, because optimizing the predicted proxy reshapes the very feedback
distribution the models are (re)trained on — a feedback-loop effect. This script surfaces those
divergences with MECHANICALLY COMPUTED flags (no statistical-significance claim, matching every prior
phaseN comparison script's headline-delta philosophy).

Closed-loop metrics (summary.json, FROZEN P20/P22 key names — CONTRACTS §4):
  reward_per_impression       <- metrics.reward_per_impression   (engagement; fallback
                                 metric_groups.engagement.reward_per_impression)
  mean_satisfaction           <- welfare.mean_immediate_satisfaction   (hidden ground-truth welfare)
  retention_7d                <- long_term.retention_7d
  sessions_per_user_per_day   <- long_term.sessions_per_user_per_day
  mean_session_utility (U_s)  <- session_health.mean_session_utility
All five are read as "higher is better" from the platform's dual engagement+welfare lens (U_s higher
= better session utility; retention/satisfaction/sessions/reward higher = better). A block absent
from a run's summary.json (e.g. a run with the P20 gates off carries no long_term block, D17) yields
an HONEST n/a cell — never a fabricated 0.

Offline-target -> closed-loop-metric ALIGNMENT (RECORDED DESIGN DECISION — the offline axis [per
observable target] and the closed-loop axis [per welfare/engagement metric] are NOT identical, so
the divergence flag pairs each learned target with the single MOST-RELATED closed-loop metric whose
delta should move WITH the offline predictiveness if it translated to a closed-loop gain):
  completed/watch_ratio/liked/shared/followed -> reward_per_impression   (engagement/watch signals)
  satisfaction                                -> mean_satisfaction        (the welfare axis)
  not_interested                              -> mean_satisfaction        (the §4.21 designed regret
                                                                           proxy — avoiding it is a
                                                                           welfare lever)
  session_exit                                -> mean_session_utility     (exit prediction -> U_s)
Targets outside this map still get their offline metric reported, with an "(no aligned metric)"
gap note. The mapping is documented in the report header so a reader can re-key it.

Usage
-----
    python3 scripts/phase23_gap_analysis.py \\
        --arm hand_tuned=results/phase23/hand_tuned/<resolved> \\
        --arm learned=results/phase23/learned/<resolved> \\
        --arm w_sat_100=results/phase23/w_sat_100/<resolved> ... \\
        --offline learned=results/phase23/offline-learned/training_eval.csv \\
        --offline w_sat_100=results/phase23/offline-w_sat_100/training_eval.csv ... \\
        --baseline hand_tuned --out results/published/phase23 [--precision 6]

    --arm LABEL=RUNDIR      a closed-loop run dir (contains summary.json). Repeatable. Include the
                            baseline AND every arm you want in the closed-loop table.
    --offline LABEL=CSV     a learned arm's offline re-eval training_eval.csv (frozen P22 §5 header).
                            Repeatable. The arms with an --offline entry are the "learned arms" that
                            get per-(arm,target) gap rows; others appear only in the closed-loop
                            table.
    --baseline LABEL        the arm every closed-loop delta is computed against (default: hand_tuned).
    --out DIR               output directory (default: results/published/phase23). Writes
                            gap_analysis.md + gap_analysis.csv.
    --precision N           significant figures for floats in the .md (default: 6).

    python3 scripts/phase23_gap_analysis.py --self-test
        Builds synthetic run-dir + training_eval.csv fixtures under a tempdir, runs the whole
        pipeline end-to-end (including a full main() invocation), asserts the aligned/divergent/
        n-a/rare cells, prints PASS/FAIL lines, cleans up, and exits. Ignores other arguments.

No third-party dependencies — plain python3 (system pre-3.10 friendly: `from __future__ import
annotations`). This script contains no simulation/training logic (D15): it only reads summary.json /
training_eval.csv and renders. Exit status: 0 on success (or --self-test pass), 1 if no usable arm
data was produced, 2 on a CLI usage error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Frozen closed-loop metric surface (summary.json; CONTRACTS §4 / P20-P22 key names) ------------
# (label, [candidate dotted paths, first hit wins], source-block name for the n/a note)
CL_METRICS: List[Tuple[str, List[str], str]] = [
    ("reward_per_impression",
     ["metrics.reward_per_impression", "metric_groups.engagement.reward_per_impression"], "metrics"),
    ("mean_satisfaction", ["welfare.mean_immediate_satisfaction"], "welfare"),
    ("retention_7d", ["long_term.retention_7d"], "long_term"),
    ("sessions_per_user_per_day", ["long_term.sessions_per_user_per_day"], "long_term"),
    ("mean_session_utility", ["session_health.mean_session_utility"], "session_health"),
]
CL_LABELS = [m[0] for m in CL_METRICS]
CL_PRETTY = {
    "reward_per_impression": "reward/impression",
    "mean_satisfaction": "mean hidden satisfaction",
    "retention_7d": "retention_7d",
    "sessions_per_user_per_day": "sessions/user/day",
    "mean_session_utility": "U_s (mean session utility)",
}

# Offline target -> aligned closed-loop metric (recorded design decision; see module docstring).
ALIGNMENT: Dict[str, str] = {
    "completed": "reward_per_impression",
    "watch_ratio": "reward_per_impression",
    "liked": "reward_per_impression",
    "shared": "reward_per_impression",
    "followed": "reward_per_impression",
    "satisfaction": "mean_satisfaction",
    "not_interested": "mean_satisfaction",
    "session_exit": "mean_session_utility",
}

# Offline baselines the learned model is judged against (P22 §5 model set; served_score included when
# it carries a comparable value). Mirrors scripts/phase22_report.py's BASELINES/verdict philosophy.
BASELINES = ["global_frequency", "per_source_frequency", "served_score"]
EVAL_REQUIRED_COLUMNS = [
    "target", "model", "split", "n_train", "n_test", "auc", "log_loss", "rmse",
    "calibration_slope", "calibration_intercept", "base_rate",
]
# P22 §7 rare-target caution threshold (expected positives in the held-out split = base_rate*n_test).
RARE_POSITIVE_THRESHOLD = 20


# --- Small shared helpers (phase22_report.py / phase15_comparison.py conventions, in miniature) ----


def warn(message: str) -> None:
    print(f"phase23_gap_analysis: {message}", file=sys.stderr)


def fmt(value, precision: int) -> str:
    """None -> 'n/a'; bool -> true/false; NaN float -> 'NaN'; else %g at `precision` sig figs."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.{precision}g}"
    return str(value)


def fmt_delta(value, precision: int) -> str:
    """Signed delta, or 'n/a'. A genuine 0.0 renders '+0'."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:+.{precision}g}"


def render_table(header: List[str], rows: List[List]) -> str:
    """Left-justified, padded, pipe-delimited markdown table (phase22_report.render_table style)."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells: List) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(header), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def _parse_float(s: Optional[str]) -> float:
    if s is None:
        return float("nan")
    s = s.strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = s.strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def dotted_get(obj: dict, path: str):
    """Follow a dotted path; return the leaf, or None if any hop is missing / not a dict."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# --- Closed-loop side (summary.json) --------------------------------------------------------------


class ArmClosedLoop:
    """One arm's closed-loop metrics + which source blocks were present (for honest n/a notes)."""

    def __init__(self, label: str, rundir: Optional[Path]):
        self.label = label
        self.rundir = rundir
        self.values: Dict[str, Optional[float]] = {m: None for m in CL_LABELS}
        self.present_blocks: set = set()
        self.loaded = False
        if rundir is not None:
            self._load(rundir)

    def _load(self, rundir: Path) -> None:
        summ = rundir / "summary.json"
        if not summ.is_file():
            warn(f"arm '{self.label}': no summary.json under {rundir} -- closed-loop metrics n/a")
            return
        try:
            with summ.open() as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            warn(f"arm '{self.label}': could not read {summ} ({exc}) -- closed-loop metrics n/a")
            return
        self.loaded = True
        for block in ("metrics", "welfare", "long_term", "session_health"):
            if isinstance(data.get(block), dict):
                self.present_blocks.add(block)
        if isinstance(dotted_get(data, "metric_groups.engagement"), dict):
            self.present_blocks.add("metrics")  # engagement fallback also satisfies reward/impression
        for label, paths, _block in CL_METRICS:
            for path in paths:
                val = dotted_get(data, path)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    self.values[label] = float(val)
                    break


def delta_of(arm: ArmClosedLoop, baseline: ArmClosedLoop, metric: str) -> Optional[float]:
    a, b = arm.values.get(metric), baseline.values.get(metric)
    if a is None or b is None:
        return None
    return a - b


# --- Offline side (training_eval.csv) -------------------------------------------------------------


class OfflineTarget:
    """The learned model's held-out metric on one target + the baseline verdict (phase22 philosophy)."""

    def __init__(self, target: str):
        self.target = target
        self.metric_type: Optional[str] = None    # "AUC" | "RMSE" | None
        self.learned: float = float("nan")
        self.baselines: Dict[str, float] = {}
        self.n_test: Optional[int] = None
        self.base_rate: float = float("nan")

    @property
    def higher_is_better(self) -> bool:
        return self.metric_type == "AUC"

    @property
    def best_baseline(self) -> Optional[float]:
        vals = [v for v in self.baselines.values() if not math.isnan(v)]
        if not vals:
            return None
        return max(vals) if self.higher_is_better else min(vals)

    @property
    def beats(self) -> Optional[str]:
        """'all' | 'some' | 'none' | None (learned metric or baselines unavailable)."""
        if self.metric_type is None or math.isnan(self.learned):
            return None
        comparable = [v for v in self.baselines.values() if not math.isnan(v)]
        if not comparable:
            return None
        won = [v for v in comparable
               if (self.learned > v if self.higher_is_better else self.learned < v)]
        if len(won) == len(comparable):
            return "all"
        if not won:
            return "none"
        return "some"

    @property
    def expected_positives(self) -> Optional[float]:
        if self.metric_type != "AUC" or math.isnan(self.base_rate) or not self.n_test:
            return None
        return self.base_rate * self.n_test

    @property
    def rare(self) -> bool:
        ep = self.expected_positives
        return ep is not None and ep < RARE_POSITIVE_THRESHOLD


def read_offline_csv(path: Path) -> Dict[str, OfflineTarget]:
    """learned-vs-baselines per target from one training_eval.csv (temporal re-eval). Never raises;
    returns {} on unreadable / bad-header input (warned)."""
    if not path.is_file():
        warn(f"offline CSV not found: {path}")
        return {}
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not set(EVAL_REQUIRED_COLUMNS).issubset(reader.fieldnames):
                warn(f"offline CSV {path} missing required frozen columns -- skipped")
                return {}
            rows = list(reader)
    except (OSError, ValueError) as exc:
        warn(f"offline CSV {path} unreadable ({exc}) -- skipped")
        return {}

    targets: Dict[str, OfflineTarget] = {}
    for row in rows:
        tname = (row.get("target") or "").strip()
        model = (row.get("model") or "").strip()
        if not tname or not model:
            continue
        ot = targets.setdefault(tname, OfflineTarget(tname))
        auc, rmse = _parse_float(row.get("auc")), _parse_float(row.get("rmse"))
        # metric type is auto-detected from which of auc/rmse the LEARNED row populates (never a
        # hardcoded target list -- same discipline as phase22_report.target_kind()).
        if model == "learned":
            ot.learned = auc if not math.isnan(auc) else rmse
            ot.metric_type = "AUC" if not math.isnan(auc) else ("RMSE" if not math.isnan(rmse) else None)
            ot.n_test = _parse_int(row.get("n_test"))
            ot.base_rate = _parse_float(row.get("base_rate"))
        elif model in BASELINES:
            # store whichever of auc/rmse is populated (aligned with the learned row's type later)
            ot.baselines[model] = auc if not math.isnan(auc) else rmse
    # Drop baselines whose metric type disagrees with the learned row (defensive; keep NaNs out).
    for ot in targets.values():
        if ot.metric_type is None:
            ot.baselines = {}
    return targets


# --- Divergence flag (mechanical) -----------------------------------------------------------------


def divergence_code(beats: Optional[str], aligned_delta: Optional[float]) -> str:
    """Short mechanical code pairing offline predictiveness with the aligned closed-loop delta."""
    if beats is None and aligned_delta is None:
        return "n/a"
    if beats is None:
        return "n/a-offline"
    if aligned_delta is None:
        return "n/a-closedloop"
    improved = aligned_delta > 0.0
    if beats == "all":
        return "ALIGNED" if improved else "DIVERGENCE"
    if beats == "none":
        return "loop-gain" if improved else "consistent-null"
    return "partial-aligned" if improved else "partial-divergence"


def divergence_sentence(target: str, arm: str, ot: Optional[OfflineTarget],
                        aligned_metric: Optional[str], aligned_delta: Optional[float],
                        precision: int) -> str:
    """Human-readable one-liner for the divergence-discussion section."""
    if aligned_metric is None:
        return f"- `{arm}` / `{target}`: no aligned closed-loop metric defined for this target."
    code = divergence_code(ot.beats if ot else None, aligned_delta)
    dstr = fmt_delta(aligned_delta, precision)
    pretty = CL_PRETTY.get(aligned_metric, aligned_metric)
    off = "n/a"
    rare = ""
    if ot is not None and ot.metric_type is not None and not math.isnan(ot.learned):
        off = f"{ot.metric_type}={fmt(ot.learned, precision)} ({ot.beats or 'no baselines'} baselines)"
        if ot.rare:
            rare = (f" [RARE TARGET: ~{ot.expected_positives:.1f} expected positives < "
                    f"{RARE_POSITIVE_THRESHOLD} — interpret the offline metric cautiously]")
    if code == "DIVERGENCE":
        body = (f"strong offline signal (learned beats ALL baselines: {off}) but the aligned "
                f"closed-loop metric **{pretty}** did NOT improve vs baseline ({dstr}) — "
                f"feedback-loop divergence: serving this proxy reshaped the distribution.")
    elif code == "ALIGNED":
        body = (f"offline-predictive ({off}) AND the aligned closed-loop metric **{pretty}** "
                f"improved vs baseline ({dstr}) — aligned, offline predictiveness tracked the loop.")
    elif code == "loop-gain":
        body = (f"NO offline signal ({off}) yet **{pretty}** improved vs baseline ({dstr}) — "
                f"closed-loop gain not attributable to this target's offline predictiveness "
                f"(other axes / loop dynamics).")
    elif code == "consistent-null":
        body = (f"no offline signal ({off}) and **{pretty}** did not improve ({dstr}) — consistent.")
    elif code == "partial-divergence":
        body = (f"offline beats SOME baselines ({off}) but **{pretty}** did not improve ({dstr}) "
                f"— partial divergence.")
    elif code == "partial-aligned":
        body = (f"offline beats SOME baselines ({off}) and **{pretty}** improved ({dstr}) "
                f"— partially aligned.")
    elif code == "n/a-offline":
        body = f"no offline re-eval available for this arm/target; **{pretty}** delta {dstr}."
    elif code == "n/a-closedloop":
        body = (f"offline {off} available but the aligned closed-loop metric **{pretty}** is "
                f"absent from this arm's summary.json (n/a).")
    else:
        body = f"insufficient data (offline and closed-loop both n/a)."
    return f"- `{arm}` / `{target}`: {body}{rare}"


# --- Rendering ------------------------------------------------------------------------------------


def render_markdown(arms: Dict[str, ArmClosedLoop], offline: Dict[str, Dict[str, OfflineTarget]],
                    baseline_label: str, precision: int) -> str:
    baseline = arms.get(baseline_label)
    learned_arms = [lbl for lbl in offline.keys()]  # arms with an offline re-eval == the learned arms
    # keep a stable, useful order: baseline first, then any non-learned arms, then learned arms
    non_learned = [lbl for lbl in arms if lbl != baseline_label and lbl not in offline]
    ordered_cl = ([baseline_label] if baseline_label in arms else []) + non_learned + \
                 [lbl for lbl in learned_arms if lbl in arms] + \
                 [lbl for lbl in learned_arms if lbl not in arms]

    P: List[str] = []
    P.append("# Phase 23 — Offline-vs-Closed-Loop Gap Analysis\n")
    P.append("Tier-5 acceptance item 5 (plan Phase 23 task 4). Generated by "
             "`scripts/phase23_gap_analysis.py` (Package B) — no simulation/training logic, only "
             "summary.json + training_eval.csv are read.\n")
    if baseline is None:
        P.append(f"> **WARNING:** baseline arm `{baseline_label}` has no closed-loop run dir — every "
                 f"closed-loop DELTA below is n/a. Pass `--arm {baseline_label}=<dir>`.\n")
    P.append("**Method.** OFFLINE = each learned arm's P22 models re-evaluated post-hoc on its OWN "
             "in-run training_log (temporal split, held-out tail) — the AUC (binary targets) or RMSE "
             "(continuous targets) of the `learned` model, judged against the frequency/served-score "
             "baselines exactly as `phase22_report.py` does (learned strictly beats each comparable "
             "baseline — a point-estimate headline, never a significance claim). CLOSED-LOOP = each "
             f"arm's summary.json outcome as a DELTA against the baseline arm `{baseline_label}` on "
             "the five frozen metrics below. Both sides are deterministic (D8/D9). All arms share an "
             "identical world (same base config + seed + streams), so every closed-loop delta is a "
             "pure ranking-policy effect.\n")
    P.append("**Closed-loop metrics (frozen P20/P22 summary.json keys).** "
             + "; ".join(f"{CL_PRETTY[l]} = `{CL_METRICS[i][1][0]}`" for i, l in enumerate(CL_LABELS))
             + ". All are read *higher = better*; a block absent from a run's summary.json gives an "
             "honest n/a cell (never a fabricated value).\n")
    P.append("**Offline→closed-loop alignment (recorded design decision).** The divergence flags "
             "pair each offline target with the single most-related closed-loop metric: "
             + "; ".join(f"`{t}`→{CL_PRETTY[m]}" for t, m in ALIGNMENT.items())
             + ". The offline axis (per observable target) and the closed-loop axis (per welfare/"
             "engagement metric) are not identical — the mapping is the interpretive bridge, stated "
             "so a reader can re-key it.\n")
    P.append("**No aggregate score (D22 / V2 TDD §6):** every metric and target is reported "
             "separately; they are never combined into one number.\n")

    # --- Section 1: closed-loop outcomes ---
    P.append("\n## 1. Closed-loop outcomes (absolute + delta vs baseline)\n")
    header = ["arm"] + [CL_PRETTY[l] for l in CL_LABELS]
    rows = []
    for lbl in ordered_cl:
        arm = arms.get(lbl)
        cells = [lbl + (" (baseline)" if lbl == baseline_label else "")]
        for l in CL_LABELS:
            if arm is None:
                cells.append("n/a")
                continue
            val = arm.values.get(l)
            if lbl == baseline_label:
                cells.append(fmt(val, precision))
            else:
                d = delta_of(arm, baseline, l) if baseline is not None else None
                cells.append(f"{fmt(val, precision)} ({fmt_delta(d, precision)})")
        rows.append(cells)
    P.append(render_table(header, rows))
    P.append("\n_Cells are `absolute (Δ vs baseline)` except the baseline row (absolute only). "
             "n/a = the source block is absent from that run's summary.json._\n")

    # --- Section 2: offline held-out metrics ---
    P.append("\n## 2. Offline held-out metrics (per learned arm × target)\n")
    if not learned_arms:
        P.append("_No `--offline` re-eval CSVs supplied — offline side unavailable._\n")
    for lbl in learned_arms:
        tgts = offline[lbl]
        P.append(f"\n### {lbl}\n")
        h2 = ["target", "metric", "learned", "best baseline", "beats", "n_test", "base_rate", "note"]
        r2 = []
        for tname in sorted(tgts):
            ot = tgts[tname]
            note = "RARE" if ot.rare else ""
            r2.append([
                tname, ot.metric_type or "n/a", fmt(ot.learned, precision),
                fmt(ot.best_baseline, precision), ot.beats or "n/a",
                ot.n_test if ot.n_test is not None else "n/a",
                fmt(ot.base_rate, precision), note,
            ])
        P.append(render_table(h2, r2) if r2 else "_(no targets parsed)_")

    # --- Section 3: the gap table ---
    P.append("\n\n## 3. Offline-vs-closed-loop gap (per learned arm × target)\n")
    h3 = ["arm", "target", "offline", "offline verdict", "aligned CL metric", "aligned Δ", "flag"]
    r3 = []
    for lbl in learned_arms:
        arm = arms.get(lbl)
        tgts = offline[lbl]
        for tname in sorted(tgts):
            ot = tgts[tname]
            aligned_metric = ALIGNMENT.get(tname)
            aligned_delta = (delta_of(arm, baseline, aligned_metric)
                             if (arm is not None and baseline is not None and aligned_metric) else None)
            off = (f"{ot.metric_type}={fmt(ot.learned, precision)}"
                   if ot.metric_type else "n/a")
            r3.append([
                lbl, tname, off, ot.beats or "n/a",
                CL_PRETTY.get(aligned_metric, "—") if aligned_metric else "—",
                fmt_delta(aligned_delta, precision) if aligned_metric else "n/a",
                divergence_code(ot.beats, aligned_delta) if aligned_metric else "no-map",
            ])
    P.append(render_table(h3, r3) if r3 else "_(no learned-arm targets to pair)_")
    P.append("\n_Flag legend: **ALIGNED** = learned beats all baselines offline AND the aligned "
             "closed-loop metric improved; **DIVERGENCE** = strong offline signal but the aligned "
             "metric did NOT improve (feedback-loop effect); loop-gain = closed-loop improved with "
             "no offline signal; consistent-null = neither; partial-* = learned beats some baselines; "
             "n/a-* = one side missing._\n")

    # --- Section 4: divergence discussion ---
    P.append("\n## 4. Divergence discussion\n")
    P.append("Mechanically computed per (arm, target); grouped by flag. The DIVERGENCE lines are the "
             "acceptance-5 headline: offline predictiveness that did not carry into the closed loop, "
             "the signature of a feedback-loop effect on the arm's own training distribution.\n")
    buckets: Dict[str, List[str]] = {}
    for lbl in learned_arms:
        arm = arms.get(lbl)
        tgts = offline[lbl]
        for tname in sorted(tgts):
            ot = tgts[tname]
            aligned_metric = ALIGNMENT.get(tname)
            aligned_delta = (delta_of(arm, baseline, aligned_metric)
                             if (arm is not None and baseline is not None and aligned_metric) else None)
            code = divergence_code(ot.beats, aligned_delta) if aligned_metric else "no-map"
            buckets.setdefault(code, []).append(
                divergence_sentence(tname, lbl, ot, aligned_metric, aligned_delta, precision))
    order = ["DIVERGENCE", "partial-divergence", "ALIGNED", "partial-aligned", "loop-gain",
             "consistent-null", "n/a-closedloop", "n/a-offline", "n/a", "no-map"]
    any_line = False
    for code in order:
        if code not in buckets:
            continue
        any_line = True
        P.append(f"\n**{code}**\n")
        P.extend(buckets[code])
    if not any_line:
        P.append("_No learned-arm/target pairs available to flag._")

    P.append("\n\n---\n")
    P.append("_Every number is a deterministic point estimate; \"beats\"/\"no signal\" means the "
             "learned model's metric strictly beats / does not beat each comparable baseline's, with "
             "no significance test (phase22_report philosophy). RARE targets (base_rate×n_test < "
             f"{RARE_POSITIVE_THRESHOLD} expected positives) are flagged — read their offline metric "
             "cautiously. n/a cells are honest absences, not zeros._\n")
    return "\n".join(P)


def write_csv(path: Path, arms: Dict[str, ArmClosedLoop],
              offline: Dict[str, Dict[str, OfflineTarget]], baseline_label: str) -> None:
    baseline = arms.get(baseline_label)
    cols = [
        "arm", "target", "offline_metric_type", "offline_learned", "offline_best_baseline",
        "offline_beats_baselines", "n_test", "base_rate", "expected_positives", "rare_target",
        "aligned_cl_metric", "aligned_cl_delta", "divergence_flag",
    ]
    for l in CL_LABELS:
        cols.append(f"cl_{l}")
        cols.append(f"cl_{l}_delta")

    def num(v):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else repr(v)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for lbl in offline.keys():
            arm = arms.get(lbl)
            tgts = offline[lbl]
            for tname in sorted(tgts):
                ot = tgts[tname]
                aligned_metric = ALIGNMENT.get(tname)
                aligned_delta = (delta_of(arm, baseline, aligned_metric)
                                 if (arm is not None and baseline is not None and aligned_metric) else None)
                row = [
                    lbl, tname, ot.metric_type or "",
                    num(ot.learned), num(ot.best_baseline), ot.beats or "",
                    ot.n_test if ot.n_test is not None else "", num(ot.base_rate),
                    num(ot.expected_positives), "1" if ot.rare else "0",
                    aligned_metric or "", num(aligned_delta),
                    divergence_code(ot.beats, aligned_delta) if aligned_metric else "no-map",
                ]
                for l in CL_LABELS:
                    val = arm.values.get(l) if arm is not None else None
                    d = delta_of(arm, baseline, l) if (arm is not None and baseline is not None) else None
                    row.append(num(val))
                    row.append(num(d))
                w.writerow(row)


# --- CLI ------------------------------------------------------------------------------------------


def _parse_kv(items: Optional[List[str]], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in (items or []):
        label, sep, value = item.partition("=")
        if not sep or not label.strip() or not value.strip():
            warn(f"ignoring malformed --{what} '{item}' (expected LABEL=PATH)")
            continue
        out[label.strip()] = value.strip()
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 23 offline-vs-closed-loop gap analysis (Package B).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", metavar="LABEL=RUNDIR", default=[],
                    help="closed-loop run dir (has summary.json). Repeatable.")
    ap.add_argument("--offline", action="append", metavar="LABEL=EVALCSV", default=[],
                    help="learned arm's offline re-eval training_eval.csv. Repeatable.")
    ap.add_argument("--baseline", default="hand_tuned",
                    help="arm every closed-loop delta is vs (default: hand_tuned).")
    ap.add_argument("--out", default="results/published/phase23",
                    help="output dir for gap_analysis.{md,csv} (default: results/published/phase23).")
    ap.add_argument("--precision", type=int, default=6, help="sig figs for floats (default: 6).")
    ap.add_argument("--self-test", action="store_true", help="run synthetic-fixture self-test and exit.")
    return ap.parse_args(argv)


def build_and_write(arm_dirs: Dict[str, str], offline_csvs: Dict[str, str],
                    baseline_label: str, out_dir: Path, precision: int) -> int:
    arms: Dict[str, ArmClosedLoop] = {
        lbl: ArmClosedLoop(lbl, Path(d)) for lbl, d in arm_dirs.items()}
    offline: Dict[str, Dict[str, OfflineTarget]] = {}
    for lbl, csv_path in offline_csvs.items():
        parsed = read_offline_csv(Path(csv_path))
        if parsed:
            offline[lbl] = parsed
        else:
            warn(f"offline arm '{lbl}': no usable targets parsed from {csv_path}")

    if not arms and not offline:
        warn("no usable --arm or --offline data supplied -- nothing to write.")
        return 1
    if baseline_label not in arms:
        warn(f"baseline '{baseline_label}' has no --arm run dir; closed-loop deltas will be n/a.")

    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(arms, offline, baseline_label, precision)
    (out_dir / "gap_analysis.md").write_text(md)
    write_csv(out_dir / "gap_analysis.csv", arms, offline, baseline_label)
    print(f"phase23_gap_analysis: wrote {out_dir/'gap_analysis.md'} and {out_dir/'gap_analysis.csv'}")
    print(f"  arms (closed-loop): {sorted(arms)}")
    print(f"  learned arms (offline): {sorted(offline)}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    arm_dirs = _parse_kv(args.arm, "arm")
    offline_csvs = _parse_kv(args.offline, "offline")
    if not arm_dirs and not offline_csvs:
        warn("nothing to do: pass --arm LABEL=RUNDIR and/or --offline LABEL=CSV (or --self-test).")
        return 2
    return build_and_write(arm_dirs, offline_csvs, args.baseline, Path(args.out), args.precision)


# --- Self-test ------------------------------------------------------------------------------------


def _check(cond: bool, message: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {message}")
    return cond


def _write_summary(rundir: Path, *, reward=None, satisfaction=None, retention=None,
                   sessions=None, u_s=None, drop_blocks=()) -> None:
    """Write a minimal summary.json with only the blocks needed by CL_METRICS; drop_blocks omits
    a whole block (to exercise the honest-n/a path)."""
    rundir.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if "metrics" not in drop_blocks:
        data["metrics"] = {"reward_per_impression": reward}
    if "welfare" not in drop_blocks:
        data["welfare"] = {"mean_immediate_satisfaction": satisfaction}
    if "long_term" not in drop_blocks:
        data["long_term"] = {"retention_7d": retention, "sessions_per_user_per_day": sessions}
    if "session_health" not in drop_blocks:
        data["session_health"] = {"mean_session_utility": u_s}
    (rundir / "summary.json").write_text(json.dumps(data, indent=2))


def _write_eval(path: Path, rows: List[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EVAL_REQUIRED_COLUMNS)
        for r in rows:
            w.writerow(r)


def run_self_test() -> int:
    """End-to-end synthetic-fixture test: a baseline arm, an ALIGNED learned arm (strong offline
    satisfaction + positive closed-loop satisfaction delta), a DIVERGENT learned arm (strong offline
    satisfaction but negative closed-loop satisfaction delta), plus a rare no-signal target and a
    learned arm missing the long_term block (honest-n/a path). Exercises the loaders, the flag logic,
    the renderer, the CSV writer, and a full main() invocation. Prints PASS/FAIL, cleans up."""
    print("phase23_gap_analysis --self-test")
    ok = True
    nan = ""  # empty cell -> NaN

    with tempfile.TemporaryDirectory(prefix="phase23-gap-selftest-") as tmp:
        root = Path(tmp)

        # --- closed-loop fixtures ---
        # baseline: satisfaction -0.30, reward 0.40, retention 0.99, sessions 1.24, U_s -2.8
        _write_summary(root / "runs/hand_tuned", reward=0.40, satisfaction=-0.30, retention=0.99,
                       sessions=1.24, u_s=-2.80)
        # aligned learned arm: satisfaction UP (-0.20 > -0.30 => +0.10 delta), reward up
        _write_summary(root / "runs/w_sat_100", reward=0.42, satisfaction=-0.20, retention=0.99,
                       sessions=1.25, u_s=-2.70)
        # divergent learned arm: satisfaction DOWN (-0.40 < -0.30 => -0.10 delta), reward up
        _write_summary(root / "runs/w_watch_100", reward=0.45, satisfaction=-0.40, retention=0.98,
                       sessions=1.30, u_s=-3.00)
        # arm with the long_term block DROPPED (retention/sessions must render n/a)
        _write_summary(root / "runs/no_lt", reward=0.41, satisfaction=-0.25,
                       u_s=-2.75, drop_blocks=("long_term",))

        # --- offline fixtures (learned beats baselines on 'satisfaction' RMSE for both learned arms;
        #     a RARE no-signal binary target 'followed'; a binary 'completed' learned beats all) ---
        def eval_rows(sat_learned_rmse: float):
            return [
                # satisfaction (continuous RMSE, lower is better): learned beats both baselines
                ("satisfaction", "learned", "temporal", 1800, 425, nan, nan, sat_learned_rmse, 0.8, 0.5, 2.47),
                ("satisfaction", "global_frequency", "temporal", 1800, 425, nan, nan, 1.26, nan, nan, 2.47),
                ("satisfaction", "per_source_frequency", "temporal", 1800, 425, nan, nan, 1.13, 0.9, 0.0, 2.47),
                # completed (binary AUC): learned beats all
                ("completed", "learned", "temporal", 8000, 2000, 0.75, 0.55, nan, 0.9, 0.02, 0.30),
                ("completed", "global_frequency", "temporal", 8000, 2000, 0.50, 0.69, nan, 0.1, 0.28, 0.30),
                ("completed", "per_source_frequency", "temporal", 8000, 2000, 0.52, 0.68, nan, 0.2, 0.25, 0.30),
                # followed (binary AUC, RARE + no signal): learned beats none, base_rate*n_test<20
                ("followed", "learned", "temporal", 8000, 2000, 0.50, 0.08, nan, 0.5, 0.01, 0.005),
                ("followed", "global_frequency", "temporal", 8000, 2000, 0.55, 0.08, nan, 0.1, 0.01, 0.005),
                ("followed", "per_source_frequency", "temporal", 8000, 2000, 0.60, 0.08, nan, 0.1, 0.01, 0.005),
            ]
        _write_eval(root / "offline/w_sat_100/training_eval.csv", eval_rows(0.90))
        _write_eval(root / "offline/w_watch_100/training_eval.csv", eval_rows(0.95))

        # --- direct loader/logic checks ---
        base = ArmClosedLoop("hand_tuned", root / "runs/hand_tuned")
        aligned = ArmClosedLoop("w_sat_100", root / "runs/w_sat_100")
        diverg = ArmClosedLoop("w_watch_100", root / "runs/w_watch_100")
        no_lt = ArmClosedLoop("no_lt", root / "runs/no_lt")

        ok &= _check(abs((delta_of(aligned, base, "mean_satisfaction") or 0) - 0.10) < 1e-9,
                     "closed-loop delta: aligned arm satisfaction +0.10 vs baseline")
        ok &= _check(abs((delta_of(diverg, base, "mean_satisfaction") or 0) + 0.10) < 1e-9,
                     "closed-loop delta: divergent arm satisfaction -0.10 vs baseline")
        ok &= _check(no_lt.values["retention_7d"] is None and "long_term" not in no_lt.present_blocks,
                     "honest n/a: dropped long_term block -> retention_7d is None")

        off_a = read_offline_csv(root / "offline/w_sat_100/training_eval.csv")
        ok &= _check(set(off_a) == {"satisfaction", "completed", "followed"},
                     f"offline loader: discovers all three targets (got {sorted(off_a)})")
        ok &= _check(off_a["satisfaction"].metric_type == "RMSE" and off_a["satisfaction"].beats == "all",
                     "offline: satisfaction RMSE, learned beats all baselines")
        ok &= _check(off_a["completed"].metric_type == "AUC" and off_a["completed"].beats == "all",
                     "offline: completed AUC, learned beats all baselines")
        ok &= _check(off_a["followed"].beats == "none" and off_a["followed"].rare,
                     "offline: followed is no-signal AND rare (base_rate*n_test < 20)")

        # flag logic
        ok &= _check(divergence_code("all", 0.10) == "ALIGNED", "flag: beats-all + positive Δ => ALIGNED")
        ok &= _check(divergence_code("all", -0.10) == "DIVERGENCE", "flag: beats-all + negative Δ => DIVERGENCE")
        ok &= _check(divergence_code("none", 0.05) == "loop-gain", "flag: no-signal + positive Δ => loop-gain")
        ok &= _check(divergence_code("all", None) == "n/a-closedloop", "flag: missing closed-loop => n/a-closedloop")
        ok &= _check(divergence_code(None, 0.1) == "n/a-offline", "flag: missing offline => n/a-offline")

        # --- full main() end-to-end ---
        out_dir = root / "published"
        rc = main([
            "--arm", f"hand_tuned={root/'runs/hand_tuned'}",
            "--arm", f"w_sat_100={root/'runs/w_sat_100'}",
            "--arm", f"w_watch_100={root/'runs/w_watch_100'}",
            "--arm", f"no_lt={root/'runs/no_lt'}",
            "--offline", f"w_sat_100={root/'offline/w_sat_100/training_eval.csv'}",
            "--offline", f"w_watch_100={root/'offline/w_watch_100/training_eval.csv'}",
            "--baseline", "hand_tuned", "--out", str(out_dir), "--precision", "6",
        ])
        ok &= _check(rc == 0, "main(): returns 0 on the full fixture")
        md_path, csv_path = out_dir / "gap_analysis.md", out_dir / "gap_analysis.csv"
        ok &= _check(md_path.is_file() and csv_path.is_file(), "main(): wrote gap_analysis.{md,csv}")

        md = md_path.read_text()
        ok &= _check("DIVERGENCE" in md, "report: contains a DIVERGENCE flag (w_watch_100/satisfaction)")
        ok &= _check("feedback-loop divergence" in md, "report: divergence-discussion prose present")
        ok &= _check("n/a" in md, "report: honest n/a cells present (dropped long_term block)")
        ok &= _check("RARE" in md, "report: rare-target caveat present (followed)")

        with csv_path.open() as fh:
            rd = list(csv.DictReader(fh))
        by = {(r["arm"], r["target"]): r for r in rd}
        ok &= _check(("w_watch_100", "satisfaction") in by
                     and by[("w_watch_100", "satisfaction")]["divergence_flag"] == "DIVERGENCE",
                     "csv: w_watch_100/satisfaction row flagged DIVERGENCE")
        ok &= _check(by[("w_sat_100", "satisfaction")]["divergence_flag"] == "ALIGNED",
                     "csv: w_sat_100/satisfaction row flagged ALIGNED")
        ok &= _check(abs(float(by[("w_sat_100", "satisfaction")]["aligned_cl_delta"]) - 0.10) < 1e-9,
                     "csv: aligned_cl_delta column = +0.10 for w_sat_100/satisfaction")
        ok &= _check(by[("w_sat_100", "followed")]["rare_target"] == "1",
                     "csv: rare_target flag set for the followed target")

        # CLI usage-error path (no args, no self-test)
        ok &= _check(main([]) == 2, "main(): CLI usage error (no --arm/--offline) returns 2")

    print("\n" + ("phase23_gap_analysis --self-test: ALL PASS" if ok
                  else "phase23_gap_analysis --self-test: SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
