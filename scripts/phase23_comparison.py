#!/usr/bin/env python3
"""phase23_comparison.py — Phase 23 multi-objective closed-loop comparison + frontier report
(V2 TDD §4.21, Tier 5 acceptance [closed-loop half], §6, §10 item 8; docs/design/
P23-CONTRACTS.md §4/§6, package C's own spec).

Renders one all-arms comparison report across the Phase 23 closed-loop experiment matrix (package
B, scripts/run_phase23_experiment.sh, contracts §4): a learned multi-objective ranker vs. hand-tuned
and semantic baselines, plus a satisfaction<->watch weight-vector sweep (arms: hand_tuned, semantic,
learned(=w_balanced), learned_survey_off, w_sat_100, w_sat_70, w_watch_70, w_watch_100,
w_watch_100_noexit -- contracts §4, 9 distinct run directories, `learned == w_balanced`), as
`comparison.md` + `.csv` under `results/published/phase23` (default).

This script reads each arm's `summary.json` / `config.json` / `explanation_sample.json` and writes
tables; it contains no simulation logic (design decision D15, matching every phaseN_comparison.py /
phase21_scenarios.py / phase22_report.py in this repo). Arms render in COMMAND-LINE ORDER (contracts
§4's arm set/count is fixed for a real Phase 23 run, but this script imposes no fixed ARM_ORDER of
its own -- same design as phase21_scenarios.py, since a caller may reasonably want a subset or a
different arm ordering while iterating pre-integration); a missing/unreadable arm still gets a row,
with every cell reported as unavailable rather than the row being silently dropped (same contract as
every prior phaseN_comparison.py/phase21_scenarios.py).

Reads, by FROZEN name only (no candidate-path guessing):
  - The four V2 §6 metric groups (engagement / hidden welfare / session health / recommendation
    quality) + `long_term` (P20 gate, contracts §5's 10-key list) -- verified directly against this
    checkout's `src/evaluation/results_writer.cpp` at HEAD 0de6f9a, unchanged since
    phase21_scenarios.py's own verification (this package re-checked rather than assumed).
  - `learned_models` (docs/design/P23-CONTRACTS.md §3, THIS PHASE's new frozen block) -- exact keys
    `configured, retrain_count, final_version, total_retrain_wall_ms, mean_n_train_rows,
    fallback_request_share, note`. NOT YET EMITTED by any committed code in this checkout at
    authoring time (package A had not landed when this script was written -- verified: grepping
    learned_ranker/learned_models/retraining_log/explanation_sample across src/include/apps/scripts
    /configs finds nothing outside docs/design/P23-CONTRACTS.md itself) -- this script is tooling
    built against the FROZEN CONTRACT TEXT for the integrator to run post-merge (contracts §7), the
    same posture phase20_comparison.py had against contracts §5 before package B landed. Absent for
    hand_tuned/semantic BY DESIGN (contracts §3: "emitted only when learned_ranker on") -- reported
    as a distinct `n/a (non-learned arm)` cell rather than a generic block-absent message, so a
    reader immediately knows this is expected, not a bug.
  - `learning_v2.value_weights` from each arm's RESOLVED `config.json` (contracts §1) -- for the
    FRONTIER table's w_satisfaction/w_watch columns. GATED ON `learning_v2.learned_ranker` being
    `true` in that SAME config.json, not on mere key presence: config.json is unconditionally
    written IN FULL regardless of any gate (D12, verified directly against
    `src/infrastructure/config.cpp`'s `LearningV2Config` to_json -- it has no `if (configured)`
    guard, unlike summary.json's gated blocks), so hand_tuned/semantic's config.json will ALSO
    resolve some value_weights (the config.hpp class defaults) even though that arm's ranker never
    reads them -- reporting those numbers in the frontier table would misleadingly suggest a
    hand-tuned/semantic arm sits somewhere on the learned satisfaction<->watch axis. Non-learned
    arms render `n/a (non-learned arm)` for both columns instead.
  - `explanation_sample.json` (contracts §6: package A's own choice of where/how to surface the
    V1 §14.4-parity explanation output; "if only in-memory, A exposes a tiny
    `--dump-explanation-sample` on simulate or writes `explanation_sample.json` per run under the
    gate -- A DECIDES and documents; C reads whatever A lands, discovered from A's committed code,
    NOT guessed"). Since package A had not landed ANY of this when this script was written (same
    authoring-time state as above), this script treats the file's contents as COMPLETELY OPAQUE
    JSON: it looks for `<arm-dir>/explanation_sample.json` in arm order, and from the FIRST one
    found and parseable, extracts ONE entry defensively -- a bare JSON list (`obj[0]`), a dict
    wrapping a list under any key (that list's `[0]`), or (if neither shape matches) the bare dict
    itself, treated as an already-flat single entry -- and pretty-prints it verbatim as fenced JSON.
    It never asserts a specific key is present (that is package A's own "explanation well-formed"
    test, contracts §5's test list, not this report's job) and never crashes on a missing file, a
    malformed one, or an unrecognized shape.

ONE FLAGGED B-COORDINATION GAP (not silently adapted around, per this package's instructions): this
script's OWN comparison.csv/.md never reads package B's `gap_analysis.csv` (that is
`scripts/plot_results.py`'s `plot_offline_closedloop_gap()`, a separate deliverable) -- see that
function's own docstring/module-section comment in plot_results.py for the specifics of that gap
(package B's `scripts/phase23_gap_analysis.py` did not exist in this tree at authoring time, so its
column names are read there by defensive candidate search, not a frozen header). This script's own
required inputs (`summary.json`/`config.json`/`explanation_sample.json`) are ALL either already-
frozen V1/V2 schema (verified against results_writer.cpp) or this phase's own contracts §1/§3 frozen
names -- there is no analogous coordination gap for comparison.py itself.

Usage
-----
    python3 scripts/phase23_comparison.py \\
        --arm hand_tuned=results/phase23/hand_tuned/<experiment-id> \\
        --arm semantic=results/phase23/semantic/<experiment-id> \\
        --arm learned=results/phase23/learned/<experiment-id> \\
        --arm learned_survey_off=results/phase23/learned_survey_off/<experiment-id> \\
        --arm w_sat_100=results/phase23/w_sat_100/<experiment-id> \\
        --arm w_sat_70=results/phase23/w_sat_70/<experiment-id> \\
        --arm w_watch_70=results/phase23/w_watch_70/<experiment-id> \\
        --arm w_watch_100=results/phase23/w_watch_100/<experiment-id> \\
        --arm w_watch_100_noexit=results/phase23/w_watch_100_noexit/<experiment-id> \\
        [--out results/published/phase23] [--precision 6] [--verdict "TEXT"]

    Any number of `--arm LABEL=DIR` may be repeated (contracts §4's 9 arms, or any subset/superset
    while iterating pre-integration); arms render in the order given on the command line. Writes
    `<out>/comparison.md` + `.csv`.

    python3 scripts/phase23_comparison.py --self-test
        Builds synthetic fixture dirs (summary.json + config.json + explanation_sample.json in a
        handful of shapes) under a tempdir, runs the loaders/renderers end-to-end (including a full
        in-process `main()` call), asserts the outputs contain expected cells, prints PASS/FAIL
        lines, cleans up, and exits. Ignores every other argument.

    (No third-party dependencies -- plain python3 is enough, same as every phaseN_comparison.py;
    only scripts/plot_results.py's additions need the uv/3.12 environment.)

Exit status: 0 if at least one arm had a readable summary.json (or --self-test passed), 1 if none
did, 2 on a CLI usage error (e.g. missing --arm without --self-test).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Make sibling-module import work regardless of CWD (scripts/ is not a package -- these scripts are
# plain, independently-runnable files per D15/TDD §22, the sys.path-relative import pattern
# scripts/phase19/20/21_comparison.py established).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase15_comparison as p15  # noqa: E402  (see sys.path.insert above)

NA_ARM = "n/a (no readable summary.json)"
NA_NON_LEARNED = "n/a (non-learned arm)"
DEFAULT_VERDICT = "VERDICT: integrator judgement pending"

# --- Report column groups: (column key, header, exact summary.json path). The first four groups
# were verified directly against src/evaluation/results_writer.cpp (HEAD 0de6f9a, re-checked by
# this package rather than assumed) -- the same V2 §6 groups phase21_scenarios.py already verified
# and reports unchanged. `learned_models` is THIS PHASE's new frozen block (contracts §3). --------

ENGAGEMENT_COLUMNS: list = [
    ("impressions", "impressions", ("counts", "impressions")),
    ("requests", "requests", ("counts", "requests")),
    ("mean_watch_seconds", "mean watch seconds", ("metrics", "mean_watch_seconds")),
    ("completion_rate", "completion rate", ("metrics", "completion_rate")),
    ("like_rate", "like rate", ("metrics", "like_rate")),
    ("share_rate", "share rate", ("metrics", "share_rate")),
    ("follow_rate", "follow rate", ("metrics", "follow_rate")),
    ("comment_rate", "comment rate", ("metric_groups", "engagement", "comment_rate")),
    ("save_rate", "save rate", ("metric_groups", "engagement", "save_rate")),
    ("profile_visit_rate", "profile-visit rate", ("metric_groups", "engagement", "profile_visit_rate")),
    ("reward_per_impression", "reward/impression", ("metrics", "reward_per_impression")),
]

WELFARE_COLUMNS: list = [
    ("mean_immediate_satisfaction", "mean hidden satisfaction",
     ("welfare", "mean_immediate_satisfaction")),
    ("mean_regret", "mean hidden regret", ("welfare", "mean_regret")),
    ("satisfaction_per_minute", "satisfaction per minute", ("welfare", "satisfaction_per_minute")),
    ("harmful_fatigue", "harmful fatigue", ("welfare", "harmful_fatigue")),
    ("platform_trust", "platform trust", ("welfare", "platform_trust")),
]

SESSION_HEALTH_COLUMNS: list = [
    ("sessions", "sessions (closed)", ("session_health", "sessions")),
    ("mean_session_utility", "mean session utility U_s", ("session_health", "mean_session_utility")),
    ("early_failure_exit_rate", "early-failure exit rate",
     ("session_health", "early_failure_exit_rate")),
    ("natural_completion_rate", "natural completion rate",
     ("session_health", "natural_completion_rate")),
    ("harmful_fatigue_mean", "harmful fatigue (mean)", ("session_health", "harmful_fatigue_mean")),
    ("exit_share_satisfied", "exit share: satisfied",
     ("session_health", "exit_type_shares", "satisfied")),
    ("exit_share_fatigue", "exit share: fatigue", ("session_health", "exit_type_shares", "fatigue")),
    ("exit_share_regret", "exit share: regret", ("session_health", "exit_type_shares", "regret")),
]

RECOMMENDATION_QUALITY_COLUMNS: list = [
    ("mean_true_affinity", "mean true affinity", ("metrics", "mean_true_affinity")),
    ("final_estimated_hidden_cosine", "estimated<->hidden cosine",
     ("learning", "final_estimated_hidden_cosine")),
    ("mean_unique_creators", "mean unique creators/feed", ("diversity", "mean_unique_creators")),
    ("mean_creator_hhi_per_feed", "mean creator HHI (per feed)", ("diversity", "mean_creator_hhi")),
    ("mean_intra_list_similarity", "mean intra-list similarity",
     ("diversity", "mean_intra_list_similarity")),
]

LONG_TERM_COLUMNS: list = [
    ("retention_1d", "retention (1d)", ("long_term", "retention_1d")),
    ("retention_7d", "retention (7d)", ("long_term", "retention_7d")),
    ("sessions_per_user_per_day", "sessions/user/day", ("long_term", "sessions_per_user_per_day")),
    ("satisfaction_weighted_retention", "satisfaction-weighted retention",
     ("long_term", "satisfaction_weighted_retention")),
    ("churn_rate", "churn rate", ("long_term", "churn_rate")),
    ("mean_churn_probability", "mean churn probability", ("long_term", "mean_churn_probability")),
    ("mean_final_trust", "mean final trust", ("long_term", "mean_final_trust")),
    ("mean_final_habit", "mean final habit", ("long_term", "mean_final_habit")),
    ("mean_preference_shift_from_initial", "mean pref shift from initial",
     ("long_term", "mean_preference_shift_from_initial")),
    ("mean_final_preference_entropy", "mean final preference entropy",
     ("long_term", "mean_final_preference_entropy")),
]

# Contracts §3 frozen `learned_models` keys (THIS PHASE's new block) -- absent entirely for
# hand_tuned/semantic (Arm.metric renders NA_NON_LEARNED for this group, see below).
LEARNED_MODELS_COLUMNS: list = [
    ("configured", "configured", ("learned_models", "configured")),
    ("retrain_count", "retrain count", ("learned_models", "retrain_count")),
    ("final_version", "final version", ("learned_models", "final_version")),
    ("total_retrain_wall_ms", "total retrain wall ms", ("learned_models", "total_retrain_wall_ms")),
    ("mean_n_train_rows", "mean n_train_rows/retrain", ("learned_models", "mean_n_train_rows")),
    ("fallback_request_share", "fallback request share",
     ("learned_models", "fallback_request_share")),
    ("note", "note", ("learned_models", "note")),
]

GROUPS: list = [
    ("Engagement", ENGAGEMENT_COLUMNS),
    ("Hidden welfare", WELFARE_COLUMNS),
    ("Session health", SESSION_HEALTH_COLUMNS),
    ("Recommendation quality", RECOMMENDATION_QUALITY_COLUMNS),
    ("Long-term (Phase 20)", LONG_TERM_COLUMNS),
    ("Learned models (Phase 23)", LEARNED_MODELS_COLUMNS),
]

ALL_COLUMNS: list = (
    ENGAGEMENT_COLUMNS + WELFARE_COLUMNS + SESSION_HEALTH_COLUMNS + RECOMMENDATION_QUALITY_COLUMNS
    + LONG_TERM_COLUMNS + LEARNED_MODELS_COLUMNS
)

# The frontier table's own metric subset (contracts §6 / plan Phase 23 task 3b): a focused
# re-presentation of five numbers already present in the groups above, alongside the two
# config-derived weight columns (w_satisfaction/w_watch) that are NOT in summary.json at all --
# kept as its own small table (rather than only relying on readers to cross-reference the big table
# + Arms section) since this is the phase's headline trade-off, the tabular counterpart to
# scripts/plot_results.py's plot_multiobjective_frontier headline FIGURE.
FRONTIER_METRIC_COLUMNS: list = [
    ("reward_per_impression", "reward/impression", ("metrics", "reward_per_impression")),
    ("mean_immediate_satisfaction", "mean hidden satisfaction",
     ("welfare", "mean_immediate_satisfaction")),
    ("retention_7d", "retention (7d)", ("long_term", "retention_7d")),
    ("mean_session_utility", "U_s (mean session utility)", ("session_health", "mean_session_utility")),
    ("mean_final_trust", "mean final trust", ("long_term", "mean_final_trust")),
]


# --- Arm loading -----------------------------------------------------------------------------


class Arm:
    """One arm's loaded data: summary.json + config.json + explanation_sample.json (frozen-key
    reads only where a schema is actually frozen; explanation_sample.json is read as opaque JSON,
    see module docstring)."""

    def __init__(self, label: str, directory: Optional[Path]):
        self.label = label
        self.directory = directory
        self.summary = p15._read_json(directory / "summary.json") if directory is not None else None
        self.config = p15._read_json(directory / "config.json") if directory is not None else None
        if directory is not None and self.summary is None:
            p15.warn(f"{label}: no readable summary.json under {directory}")

    @property
    def available(self) -> bool:
        return self.summary is not None

    def block_present(self, block: str) -> bool:
        return isinstance(self.summary, dict) and isinstance(self.summary.get(block), dict)

    def metric(self, path: tuple):
        """Frozen-name-exact lookup (no candidate guessing). Distinguishes: the whole arm never
        loaded (NA_ARM); the `learned_models` block specifically absent -- the expected, BY-DESIGN
        state for hand_tuned/semantic (NA_NON_LEARNED, contracts §3); any OTHER containing block
        absent (a generic, distinctly-labeled n/a); or just this one key missing from an otherwise-
        present block."""
        if not self.available:
            return NA_ARM
        value = p15._get(self.summary, *path)
        if value is not None:
            return value
        if not self.block_present(path[0]):
            if path[0] == "learned_models":
                return NA_NON_LEARNED
            return f"n/a (block '{path[0]}' absent)"
        return f"n/a (key '{'.'.join(path)}' absent)"

    def is_learned(self) -> Optional[bool]:
        """This arm's resolved `learning_v2.learned_ranker` (contracts §1), or None when
        config.json is unavailable or the key itself is absent (e.g. a pre-Phase-23 config)."""
        value = p15._get(self.config, "learning_v2", "learned_ranker")
        return value if isinstance(value, bool) else None

    def value_weights(self) -> Optional[dict]:
        """This arm's resolved `learning_v2.value_weights` dict (contracts §1), or None when this
        is not a learned arm (`is_learned()` is not True) or the block/config is unavailable. Gated
        on learned_ranker rather than mere key presence -- see module docstring: config.json is
        unconditionally written in full (D12), so a non-learned arm's config.json still resolves
        SOME value_weights (the config.hpp class defaults), which would be meaningless to report
        since that arm's ranker never reads them.
        """
        if self.is_learned() is not True:
            return None
        vw = p15._get(self.config, "learning_v2", "value_weights")
        return vw if isinstance(vw, dict) else None

    def frontier_weight(self, key: str):
        """w_satisfaction (`key="satisfaction"`) or w_watch (`key="watch"`) for the frontier table,
        or a distinctly-labeled n/a for a non-learned arm / a learned arm whose value_weights is
        missing that specific key. An explicit 0.0 (e.g. w_watch_100's watch=0 -- wait, satisfaction
        arms zero out the OTHER axis, e.g. w_sat_100 sets watch=0.0) is never mistaken for an absent
        key: only `None` (key genuinely missing) falls through to the n/a message."""
        vw = self.value_weights()
        if vw is None:
            return NA_NON_LEARNED
        value = vw.get(key)
        return value if value is not None else f"n/a (key '{key}' absent from value_weights)"

    def explanation_sample_path(self) -> Optional[Path]:
        if self.directory is None:
            return None
        path = self.directory / "explanation_sample.json"
        return path if path.exists() else None


def _parse_arm_arg(s: str) -> tuple:
    """Parses one `--arm LABEL=DIR` value (same shape as phase21_scenarios.py's own helper,
    reproduced here rather than cross-imported -- a handful of lines, matching phase22_report.py's
    precedent for not adding a cross-script import for tiny helpers). Raises ValueError (caught by
    argparse's `type=` machinery, reported as a normal usage error) when there is no '=' or either
    side is empty after stripping."""
    if "=" not in s:
        raise ValueError(f"--arm value {s!r} must be of the form LABEL=DIR")
    label, _, directory = s.partition("=")
    label, directory = label.strip(), directory.strip()
    if not label or not directory:
        raise ValueError(f"--arm value {s!r} must be of the form LABEL=DIR (both non-empty)")
    return label, directory


def load_arms(arm_pairs: list) -> dict:
    """Builds the label -> Arm map in COMMAND-LINE ORDER (contracts §4's arm set is fixed for a
    real Phase 23 run, but this script imposes no fixed ARM_ORDER of its own -- same rationale as
    phase21_scenarios.py's load_arms). A repeated label overwrites the earlier one (warned)."""
    arms: dict = {}
    for label, directory in arm_pairs:
        if label in arms:
            p15.warn(f"--arm {label}=... given more than once; using the last occurrence")
        arms[label] = Arm(label, Path(directory))
    return arms


# --- explanation_sample.json: opaque-JSON single-entry extraction (contracts §6) -----------------


def _first_explanation_entry(obj):
    """Best-effort extraction of ONE entry from an arbitrarily-shaped explanation_sample.json
    (package A's schema is undiscovered/opaque at authoring time -- see module docstring). Handles:
    a bare JSON list (-> obj[0]); a dict wrapping a list under ANY key (-> that list's [0], the
    first such key found in insertion order); or a bare dict with no list-valued key at all (-> the
    dict itself, treated as an already-flat single entry). Returns None for an empty list, a dict
    with no usable content, or any other JSON shape (a bare number/string/bool/null)."""
    if isinstance(obj, list):
        return obj[0] if obj else None
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, list) and value:
                return value[0]
        return obj if obj else None
    return None


def find_explanation_sample(arms: dict):
    """Returns (label, path, entry) for the FIRST arm (command-line order) with a readable,
    non-empty `explanation_sample.json`, or (None, None, None) if no arm has one -- mirrors
    phase21_scenarios.py's find_description() precedent (first-arm-with-it wins). A present-but-
    unparseable file, or one with no usable entry, is skipped (warned) rather than treated as fatal
    -- the search continues to the next arm."""
    for arm in arms.values():
        path = arm.explanation_sample_path()
        if path is None:
            continue
        obj = p15._read_json(path)
        if obj is None:
            p15.warn(f"{arm.label}: explanation_sample.json exists but failed to parse; skipping")
            continue
        entry = _first_explanation_entry(obj)
        if entry is None:
            p15.warn(f"{arm.label}: explanation_sample.json present but no usable entry found "
                     f"(empty list/dict)")
            continue
        return arm.label, path, entry
    return None, None, None


# --- Rendering -----------------------------------------------------------------------------------


def render_table(header: list, rows: list) -> str:
    """Left-justified, padded, pipe-delimited markdown table (same visual style as
    phase15_comparison.py's render_markdown_table / phase21_scenarios.py's render_table, reproduced
    here rather than cross-imported -- see _parse_arm_arg's docstring for the same reasoning)."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells: list) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(header), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def render_group_table(arms: dict, columns: list, precision: int) -> str:
    header = ["arm"] + [h for _k, h, _p in columns]
    rows = []
    for arm in arms.values():
        rows.append([arm.label] + [p15.fmt(arm.metric(p), precision) for _k, _h, p in columns])
    return render_table(header, rows)


def render_frontier_table(arms: dict, precision: int) -> str:
    header = ["arm", "w_satisfaction", "w_watch"] + [h for _k, h, _p in FRONTIER_METRIC_COLUMNS]
    rows = []
    for arm in arms.values():
        row = [arm.label, p15.fmt(arm.frontier_weight("satisfaction"), precision),
               p15.fmt(arm.frontier_weight("watch"), precision)]
        row.extend(p15.fmt(arm.metric(p), precision) for _k, _h, p in FRONTIER_METRIC_COLUMNS)
        rows.append(row)
    return render_table(header, rows)


def render_retraining_table(arms: dict, precision: int) -> str:
    """Derived from the `learned_models` block only (this package's brief), plus one derived stat
    (`mean_wall_ms_per_retrain = total_retrain_wall_ms / retrain_count`, computed here -- not
    itself a contracts §3 key). `retraining_log.csv`'s own per-retrain-event rows are NOT parsed for
    this table (see Notes section / module docstring)."""
    header = ["arm", "configured", "retrain count", "final version", "total retrain wall ms",
              "mean wall ms/retrain", "mean n_train_rows", "fallback request share", "note"]
    rows = []
    for arm in arms.values():
        retrain_count = arm.metric(("learned_models", "retrain_count"))
        total_wall = arm.metric(("learned_models", "total_retrain_wall_ms"))
        mean_wall = "n/a"
        if p15._is_num(retrain_count) and p15._is_num(total_wall):
            mean_wall = (total_wall / retrain_count) if retrain_count > 0 else "n/a (zero retrains)"
        rows.append([
            arm.label,
            p15.fmt(arm.metric(("learned_models", "configured")), precision),
            p15.fmt(retrain_count, precision),
            p15.fmt(arm.metric(("learned_models", "final_version")), precision),
            p15.fmt(total_wall, precision),
            p15.fmt(mean_wall, precision),
            p15.fmt(arm.metric(("learned_models", "mean_n_train_rows")), precision),
            p15.fmt(arm.metric(("learned_models", "fallback_request_share")), precision),
            p15.fmt(arm.metric(("learned_models", "note")), precision),
        ])
    return render_table(header, rows)


def render_arms_section(arms: dict) -> list:
    lines = ["## Arms and data availability\n"]
    block_names = ["welfare", "session_health", "long_term", "learned_models"]
    for arm in arms.values():
        lines.append(f"- **{arm.label}**")
        lines.append(f"  - directory: `{arm.directory}`")
        if not arm.available:
            lines.append(f"  - status: NOT AVAILABLE ({NA_ARM})")
            continue
        lines.append(f"  - resolved `learning_v2.learned_ranker`: {arm.is_learned()}")
        vw = arm.value_weights()
        if vw is not None:
            weights_str = ", ".join(f"{k}={v:g}" for k, v in sorted(vw.items())
                                    if isinstance(v, (int, float)) and not isinstance(v, bool))
            lines.append(f"  - `value_weights`: {weights_str or '(none numeric)'}")
        presence = ", ".join(f"{b}={arm.block_present(b)}" for b in block_names)
        lines.append(f"  - gated-block presence: {presence}")
        sample_path = arm.explanation_sample_path()
        lines.append(f"  - `explanation_sample.json`: "
                     f"{'present' if sample_path is not None else 'ABSENT'}")
    return lines


def render_explanation_section(arms: dict) -> list:
    lines = ["## Explanation sample (one served candidate, pretty-printed)\n"]
    label, path, entry = find_explanation_sample(arms)
    if entry is None:
        lines.append(
            "*n/a -- no arm directory has a readable, non-empty `explanation_sample.json` "
            "(package A's contracts §6 choice for surfacing the V1 §14.4-parity explanation "
            "output). This script treats its contents as OPAQUE JSON (module docstring) and does "
            "not guess a schema when the file is absent everywhere.*\n"
        )
        return lines
    lines.append(
        f"From **{label}**'s `explanation_sample.json` (`{path}`), first entry, pretty-printed "
        f"VERBATIM -- this script does not interpret, validate, or assert against its schema "
        f"(that is package A's own \"explanation well-formed\" test, contracts §5):\n"
    )
    lines.append("```json")
    lines.append(json.dumps(entry, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    return lines


def render_notes() -> list:
    return [
        "## Notes\n",
        "- **No aggregate score** (D22 / V2 TDD §6): every group above (engagement, hidden "
        "welfare, session health, recommendation quality, long-term, learned models) is reported "
        "SEPARATELY; this script never combines them into one number, and neither does "
        "`scripts/plot_results.py`'s `plot_multiobjective_frontier` (a 2-axis scatter with a "
        "third size-encoded dimension, not a blended score).\n",
        "- **`learned == w_balanced`** (contracts §4): the frontier sweep's \"balanced\" weight "
        "vector is the SAME run directory as the `learned` arm (both use the config.hpp "
        "`value_weights` defaults); this report does not duplicate that row under two labels -- "
        "pass `--arm w_balanced=<same dir as learned>` explicitly if a rendered `w_balanced` row "
        "is wanted alongside `learned`.\n",
        "- **Frozen-schema reads only**: every key this script reads is quoted verbatim from "
        "docs/design/P23-CONTRACTS.md §1 (`value_weights`) / §3 (`learned_models`), or is a V2 §6 "
        "group / `long_term` key already verified against `src/evaluation/results_writer.cpp` by "
        "phase20/21_comparison.py and re-checked by this package (module docstring) -- no "
        "candidate-path guessing. A key genuinely missing from an otherwise-present block renders "
        "as a clearly labeled n/a, never a crash or a silently-guessed alternate key.\n",
        "- **`explanation_sample.json` is opaque** (contracts §6): this script pretty-prints one "
        "entry verbatim and asserts nothing about its internal shape -- see module docstring and "
        "the Explanation-sample section above.\n",
        "- **`retraining_log.csv`** (contracts §3 frozen header: `version,sim_time_seconds,"
        "n_train_rows,wall_ms,targets_trained`) is a per-run companion artifact this report does "
        "not parse row-by-row (the retraining summary table above is derived from summary.json's "
        "`learned_models` block only, per this package's brief) -- it is available for a deeper "
        "per-retrain-event timeline if a future report wants one.\n",
        "- **Concurrency-contention caveat**: if arms were run CONCURRENTLY (contracts §4 caps "
        "package B's own script at 3 concurrent), wall-clock/timing numbers (incl. "
        "`total_retrain_wall_ms`, measured by `steady_clock` OUTSIDE simulated time per contracts "
        "§3) carry cache and memory-bandwidth contention; every other number in this report is "
        "deterministic (rng/clock-free, D8/D9) and unaffected by run mode.\n",
    ]


def render_markdown(arms: dict, verdict_text: str, precision: int) -> str:
    parts = ["# Phase 23 -- Learned Multi-Objective Ranking in the Loop\n"]
    parts.append(
        "V2 TDD §4.21, Tier 5 acceptance (closed-loop half), §6, §10 item 8 (docs/design/"
        "P23-CONTRACTS.md §4/§6): a learned multi-objective ranker (`hnsw_learned_ranker`) "
        "compared closed-loop against hand-tuned (`hnsw_ranker`) and semantic-similarity "
        "baselines, plus a satisfaction<->watch weight-vector sweep exposing the "
        "engagement/welfare/retention trade-off frontier. Generated by "
        "`scripts/phase23_comparison.py`.\n"
    )

    parts.append("## Method\n")
    parts.append(
        "**Identical-world discipline** (contracts §4 / D8/D17): every arm is meant to run on the "
        "SAME base world -- `configs/realism-medium-retention.json` + "
        "`learning_v2.training_log=true` + `learning_v2.survey.enabled=true`, seed 42, event mode "
        "-- so arm-vs-arm differences are attributable to the ranking policy / weight vector, "
        "never to a different user/content/interaction stream. `scripts/run_phase23_experiment.sh` "
        "(package B) generates each arm's config via an additive JSON patch over that base; this "
        "script only reads results, never re-simulates (design decision D15, matching every "
        "phaseN_comparison.py in this repo).\n\n"
        "**No aggregate score** (D22 / V2 TDD §6): see Notes below -- every metric group is "
        "reported separately; no single number is ever computed to rank the arms.\n"
    )

    parts.extend(render_arms_section(arms))
    parts.append("")

    parts.append("## Metric groups\n")
    for label, columns in GROUPS:
        parts.append(f"### {label}\n")
        parts.append(render_group_table(arms, columns, precision))
        parts.append("")

    parts.append("## Multi-objective frontier\n")
    parts.append(
        "`w_satisfaction`/`w_watch` are read from each arm's RESOLVED `config.json` "
        "`learning_v2.value_weights` (contracts §1), gated on `learning_v2.learned_ranker` being "
        "true (see module docstring for why mere key presence is not enough); non-learned arms "
        f"(`hand_tuned`/`semantic`) render `{NA_NON_LEARNED}` for both. The remaining columns are "
        "the same headline numbers `scripts/plot_results.py`'s `plot_multiobjective_frontier` "
        "plots (engagement vs. hidden satisfaction, marker size ~ retention_7d), plus U_s and "
        "mean final trust for a fuller welfare/retention picture in table form.\n"
    )
    parts.append(render_frontier_table(arms, precision))
    parts.append("")

    parts.append("## Retraining summary\n")
    parts.append(
        "Derived from each arm's `learned_models` summary.json block (contracts §3 frozen keys); "
        f"`{NA_NON_LEARNED}` for hand_tuned/semantic (the block is emitted only when "
        "`learned_ranker` is on). `mean wall ms/retrain` is this script's own derived "
        "`total_retrain_wall_ms / retrain_count` (not itself a contracts §3 key).\n"
    )
    parts.append(render_retraining_table(arms, precision))
    parts.append("")

    parts.extend(render_explanation_section(arms))

    parts.append("## Verdict\n")
    parts.append(f"{verdict_text}\n")

    parts.extend(render_notes())
    return "\n".join(parts) + "\n"


def write_csv(path: Path, arms: dict, precision: int) -> None:
    header = ["arm", "w_satisfaction", "w_watch"] + [h for _k, h, _p in ALL_COLUMNS]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for arm in arms.values():
            row = [arm.label, p15.fmt(arm.frontier_weight("satisfaction"), precision),
                   p15.fmt(arm.frontier_weight("watch"), precision)]
            for _k, _h, col_path in ALL_COLUMNS:
                row.append(p15.fmt(arm.metric(col_path), precision))
            writer.writerow(row)


# --- CLI -------------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Phase 23 learned-multi-objective-ranking closed-loop experiment "
                    "(docs/design/P23-CONTRACTS.md §6) as comparison.{md,csv}.",
    )
    parser.add_argument("--arm", action="append", dest="arms", default=None, type=_parse_arm_arg,
                        metavar="LABEL=DIR",
                        help="one arm as LABEL=DIR; repeat for each arm (contracts §4: hand_tuned, "
                             "semantic, learned, learned_survey_off, w_sat_100, w_sat_70, "
                             "w_watch_70, w_watch_100, w_watch_100_noexit). Required at least "
                             "once, unless --self-test.")
    parser.add_argument("--out", type=Path, default=Path("results/published/phase23"),
                        help="output directory for comparison.{md,csv} "
                             "(default: results/published/phase23)")
    parser.add_argument("--precision", type=int, default=6,
                        help="significant figures for floats (default: 6)")
    parser.add_argument("--verdict", default=None,
                        help=f"verdict text (default: {DEFAULT_VERDICT!r})")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="run an in-script synthetic-fixture smoke test end-to-end, print "
                             "PASS/FAIL lines, and exit (ignores every other argument; no "
                             "committed fixture files)")
    args = parser.parse_args(argv)
    if not args.self_test and not args.arms:
        parser.error("at least one --arm LABEL=DIR is required (unless --self-test)")
    return args


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    if args.self_test:
        return run_self_test()

    arms = load_arms(args.arms)
    if not any(arm.available for arm in arms.values()):
        p15.warn("no readable summary.json in any given arm")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    verdict_text = args.verdict.strip() if args.verdict else DEFAULT_VERDICT

    csv_path = args.out / "comparison.csv"
    write_csv(csv_path, arms, args.precision)

    md_path = args.out / "comparison.md"
    md_path.write_text(render_markdown(arms, verdict_text, args.precision))

    print(f"phase23_comparison: wrote {csv_path}")
    print(f"phase23_comparison: wrote {md_path}")
    return 0


# --- Self-test -------------------------------------------------------------------------------


def _check(cond: bool, message: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {message}")
    return cond


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _base_summary(learned: bool, reward: float, satisfaction: float, retention_7d: float,
                  u_s: float, mean_final_trust: float, retrain_count: int = 5,
                  fallback_share: float = 0.02) -> dict:
    s = {
        "counts": {"impressions": 12000, "requests": 3000},
        "metrics": {"mean_watch_seconds": 12.0, "completion_rate": 0.4, "like_rate": 0.1,
                   "share_rate": 0.02, "follow_rate": 0.01, "reward_per_impression": reward,
                   "mean_true_affinity": 0.55},
        "metric_groups": {"engagement": {"comment_rate": 0.01, "save_rate": 0.02,
                                         "profile_visit_rate": 0.03}},
        "learning": {"final_estimated_hidden_cosine": 0.8},
        "diversity": {"mean_unique_creators": 6.0, "mean_creator_hhi": 0.3,
                     "mean_intra_list_similarity": 0.2},
        "welfare": {"mean_immediate_satisfaction": satisfaction, "mean_regret": 0.1,
                   "satisfaction_per_minute": 0.05, "harmful_fatigue": 0.03,
                   "platform_trust": 0.6},
        "session_health": {"sessions": 100, "mean_session_utility": u_s,
                          "early_failure_exit_rate": 0.1, "natural_completion_rate": 0.5,
                          "harmful_fatigue_mean": 0.05,
                          "exit_type_shares": {"satisfied": 0.5, "fatigue": 0.2, "regret": 0.05}},
        "long_term": {
            "retention_1d": 0.8, "retention_7d": retention_7d, "sessions_per_user_per_day": 1.2,
            "satisfaction_weighted_retention": 0.55, "churn_rate": 0.1,
            "mean_churn_probability": 0.12, "mean_final_trust": mean_final_trust,
            "mean_final_habit": 0.4, "mean_preference_shift_from_initial": 0.1234,
            "mean_final_preference_entropy": 1.5,
        },
    }
    if learned:
        s["learned_models"] = {
            "configured": True,
            "retrain_count": retrain_count,
            "final_version": retrain_count,
            "total_retrain_wall_ms": 1000.0 * retrain_count,
            "mean_n_train_rows": 8000,
            "fallback_request_share": fallback_share,
            "note": "in-loop SGD retrain, deterministic schedule",
        }
    return s


def _base_config(learned: bool, weights: Optional[dict] = None) -> dict:
    value_weights = {"watch": 0.30, "share": 0.15, "follow": 0.10, "satisfaction": 0.30,
                     "exit": 0.10, "regret": 0.05}
    if weights:
        value_weights.update(weights)
    return {"learning_v2": {"learned_ranker": learned, "value_weights": value_weights,
                            "retrain_every_hours": 24, "min_training_rows": 5000,
                            "retrain_epochs": 50}}


def run_self_test() -> int:
    """Synthetic-fixture smoke test (no committed fixture files). Six arms under a tempdir:
    `hand_tuned`/`semantic` (non-learned: learned_ranker=false, no learned_models block, no
    explanation_sample.json -- exercises NA_NON_LEARNED and the config-gated value_weights logic);
    `corrupt_sample` (learned, but its explanation_sample.json is deliberately INVALID JSON text --
    exercises the parse-failure skip path); `w_sat_100` (learned, satisfaction=0.60/watch=0.0 --
    the watch=0.0 case specifically checks 0 is not mistaken for "key absent"; no
    explanation_sample.json at all -- exercises the file-absent skip path); `learned` (= w_balanced,
    default value_weights, HAS a valid LIST-shaped explanation_sample.json with two entries -- the
    one this search is expected to land on, after skipping corrupt_sample and w_sat_100); `missing`
    (an empty directory -- exercises NA_ARM). Exercises the loaders/renderers directly AND a full
    in-process `main()` call (including the CLI usage-error path). Prints one PASS/FAIL line per
    check."""
    print("phase23_comparison --self-test")
    ok = True

    # --- _first_explanation_entry shape coverage (unit-level, no fixture files needed) -----------
    ok &= _check(_first_explanation_entry([{"a": 1}, {"a": 2}]) == {"a": 1},
                "_first_explanation_entry: bare list -> first element")
    ok &= _check(_first_explanation_entry([]) is None,
                "_first_explanation_entry: empty list -> None")
    ok &= _check(_first_explanation_entry({"samples": [{"a": 1}], "meta": "x"}) == {"a": 1},
                "_first_explanation_entry: dict wrapping a list under some key -> that list's [0]")
    ok &= _check(_first_explanation_entry({"a": 1, "b": 2}) == {"a": 1, "b": 2},
                "_first_explanation_entry: bare dict with no list-valued key -> the dict itself")
    ok &= _check(_first_explanation_entry({}) is None,
                "_first_explanation_entry: empty dict -> None")
    ok &= _check(_first_explanation_entry("not a dict or list") is None,
                "_first_explanation_entry: non-dict/non-list -> None")

    with tempfile.TemporaryDirectory(prefix="phase23-comparison-selftest-") as tmp:
        root = Path(tmp)

        _write_json(root / "hand_tuned" / "summary.json",
                   _base_summary(False, reward=0.20, satisfaction=0.40, retention_7d=0.45,
                                u_s=0.28, mean_final_trust=0.55))
        _write_json(root / "hand_tuned" / "config.json", _base_config(False))

        _write_json(root / "semantic" / "summary.json",
                   _base_summary(False, reward=0.15, satisfaction=0.35, retention_7d=0.40,
                                u_s=0.25, mean_final_trust=0.50))
        _write_json(root / "semantic" / "config.json", _base_config(False))

        _write_json(root / "corrupt_sample" / "summary.json",
                   _base_summary(True, reward=0.22, satisfaction=0.50, retention_7d=0.52,
                                u_s=0.31, mean_final_trust=0.60, retrain_count=3,
                                fallback_share=0.10))
        _write_json(root / "corrupt_sample" / "config.json",
                   _base_config(True, {"satisfaction": 0.42, "watch": 0.18}))
        (root / "corrupt_sample" / "explanation_sample.json").write_text("{not valid json")

        _write_json(root / "w_sat_100" / "summary.json",
                   _base_summary(True, reward=0.18, satisfaction=0.60, retention_7d=0.58,
                                u_s=0.33, mean_final_trust=0.62, retrain_count=4,
                                fallback_share=0.01))
        _write_json(root / "w_sat_100" / "config.json",
                   _base_config(True, {"satisfaction": 0.60, "watch": 0.0}))
        # deliberately NO explanation_sample.json for this arm.

        _write_json(root / "learned" / "summary.json",
                   _base_summary(True, reward=0.25, satisfaction=0.45, retention_7d=0.55,
                                u_s=0.32, mean_final_trust=0.61, retrain_count=5,
                                fallback_share=0.02))
        _write_json(root / "learned" / "config.json", _base_config(True))
        _write_json(root / "learned" / "explanation_sample.json", [
            {"predicted_watch": 0.7, "predicted_share": 0.05, "predicted_follow": 0.02,
             "predicted_satisfaction": 0.65, "predicted_exit": 0.1, "predicted_regret": 0.05,
             "learned_value": 0.42, "fallback": 0, "satisfaction_available": 1},
            {"predicted_watch": 0.3, "predicted_share": 0.01, "predicted_follow": 0.0,
             "predicted_satisfaction": 0.2, "predicted_exit": 0.4, "predicted_regret": 0.2,
             "learned_value": -0.05, "fallback": 0, "satisfaction_available": 1},
        ])

        # "missing": directory exists but carries no files at all.
        (root / "missing").mkdir(parents=True, exist_ok=True)

        arm_pairs = [
            ("hand_tuned", str(root / "hand_tuned")),
            ("semantic", str(root / "semantic")),
            ("corrupt_sample", str(root / "corrupt_sample")),
            ("w_sat_100", str(root / "w_sat_100")),
            ("learned", str(root / "learned")),
            ("missing", str(root / "missing")),
        ]
        arms = load_arms(arm_pairs)

        ok &= _check(list(arms.keys()) == [label for label, _d in arm_pairs],
                    "load_arms: preserves command-line order")
        ok &= _check(arms["hand_tuned"].available and arms["learned"].available,
                    "Arm.available: hand_tuned and learned both load")
        ok &= _check(not arms["missing"].available, "Arm.available: missing arm is unavailable")

        ok &= _check(arms["missing"].metric(("welfare", "mean_immediate_satisfaction")) == NA_ARM,
                    "Arm.metric: an unavailable arm reads as NA_ARM")
        ok &= _check(
            arms["hand_tuned"].metric(("learned_models", "retrain_count")) == NA_NON_LEARNED,
            "Arm.metric: a non-learned arm's learned_models.* reads as NA_NON_LEARNED")
        ok &= _check(arms["learned"].metric(("learned_models", "retrain_count")) == 5,
                    "Arm.metric: learned arm's learned_models.retrain_count reads correctly")

        ok &= _check(arms["hand_tuned"].is_learned() is False,
                    "Arm.is_learned: hand_tuned resolves to False")
        ok &= _check(arms["learned"].is_learned() is True,
                    "Arm.is_learned: learned resolves to True")

        ok &= _check(arms["hand_tuned"].value_weights() is None,
                    "Arm.value_weights: non-learned arm -> None (gated on learned_ranker, not "
                    "mere key presence -- D12 always resolves SOME value_weights)")
        learned_vw = arms["learned"].value_weights()
        ok &= _check(isinstance(learned_vw, dict) and learned_vw.get("satisfaction") == 0.30,
                    "Arm.value_weights: learned arm's resolved value_weights reads correctly")

        ok &= _check(arms["hand_tuned"].frontier_weight("satisfaction") == NA_NON_LEARNED,
                    "Arm.frontier_weight: non-learned arm -> NA_NON_LEARNED")
        ok &= _check(arms["w_sat_100"].frontier_weight("satisfaction") == 0.60,
                    "Arm.frontier_weight: w_sat_100's w_satisfaction reads correctly")
        ok &= _check(arms["w_sat_100"].frontier_weight("watch") == 0.0,
                    "Arm.frontier_weight: an explicit watch=0.0 is NOT mistaken for an absent key")

        label, path, entry = find_explanation_sample(arms)
        ok &= _check(label == "learned",
                    f"find_explanation_sample: skips corrupt_sample (parse failure) and "
                    f"w_sat_100 (file absent), lands on learned (got label={label!r})")
        ok &= _check(isinstance(entry, dict) and entry.get("predicted_watch") == 0.7,
                    "find_explanation_sample: returns the FIRST entry of learned's list-shaped "
                    "explanation_sample.json")

        md = render_markdown(arms, DEFAULT_VERDICT, precision=6)
        ok &= _check(NA_NON_LEARNED in md, "render_markdown: NA_NON_LEARNED cell text is present")
        ok &= _check(DEFAULT_VERDICT in md, "render_markdown: default verdict text present")
        ok &= _check('"predicted_watch": 0.7' in md,
                    "render_markdown: the explanation sample's fenced JSON is present verbatim")
        ok &= _check(all(f"### {label}" in md for label, _cols in GROUPS),
                    "render_markdown: all six group section headers present")
        ok &= _check("## Multi-objective frontier" in md and "## Retraining summary" in md,
                    "render_markdown: frontier and retraining sections present")

        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "comparison.csv"
        write_csv(csv_path, arms, precision=6)
        with csv_path.open() as fh:
            csv_rows = list(csv.reader(fh))
        ok &= _check(len(csv_rows) == 1 + len(arms),
                    f"write_csv: header + {len(arms)} arm rows (got {len(csv_rows)} total rows)")
        ok &= _check(csv_rows[0][:3] == ["arm", "w_satisfaction", "w_watch"],
                    "write_csv: leading arm/w_satisfaction/w_watch columns")

        ok &= _check(_parse_arm_arg("learned=/tmp/x") == ("learned", "/tmp/x"),
                    "_parse_arm_arg: parses LABEL=DIR")
        try:
            _parse_arm_arg("no-equals-sign")
            ok &= _check(False, "_parse_arm_arg: raises ValueError without '='")
        except ValueError:
            ok &= _check(True, "_parse_arm_arg: raises ValueError without '='")

        # Full in-process CLI/main() round trip, custom --verdict text.
        main_out = root / "main-out"
        cli_args = ["--out", str(main_out), "--verdict", "custom verdict text",
                   "--precision", "4"]
        for label, directory in arm_pairs:
            cli_args += ["--arm", f"{label}={directory}"]
        rc = main(cli_args)
        ok &= _check(rc == 0, "main(): exits 0 when at least one arm is available")
        md_path = main_out / "comparison.md"
        csv_path2 = main_out / "comparison.csv"
        ok &= _check(md_path.exists() and csv_path2.exists(),
                    "main(): writes comparison.{md,csv}")
        if md_path.exists():
            md_text = md_path.read_text()
            ok &= _check("custom verdict text" in md_text,
                        "main(): custom --verdict text lands in the rendered report")

        rc_missing_args = main([])
        ok &= _check(rc_missing_args == 2,
                    f"main(): no --arm given (no --self-test) exits 2 (got {rc_missing_args})")

    print(f"phase23_comparison --self-test: {'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
