#!/usr/bin/env python3
"""phase22_report.py — Phase 22 offline-evaluation report (V2 TDD §4.19-4.20/§4.22, Tier-5
acceptance 1 [offline half]; docs/design/P22-CONTRACTS.md §6 [this package's own spec] + §5).

Renders `training_eval.csv` (contracts §5 frozen header: `target,model,split,n_train,n_test,auc,
log_loss,rmse,calibration_slope,calibration_intercept,base_rate`, one row per (target, model,
split) — models `learned,global_frequency,per_source_frequency,served_score`) from one or two
`train_models` runs (contracts §6: one invocation per split, `temporal` and `user_disjoint`,
scripts/run_phase22_logworld.sh) into `<out>/offline_eval.md`:

  - a method preamble (identical-world discipline: ONE pinned log-world run, seed 42; the two
    split definitions);
  - one section PER TARGET, with a single flat table whose columns are grouped "SPLIT: metric"
    for every split found in the input data (both splits side by side in one table when two
    `--eval` files are given) — rows are the four frozen models, columns are n_train/n_test/
    base_rate plus AUC/log-loss/calibration (binary targets) or RMSE/calibration (continuous
    targets), auto-detected per target from which of auc/rmse is actually populated (never a
    hardcoded eight-target list — see the module-level note on target-name ambiguity below);
  - an honest, MECHANICALLY COMPUTED per-target verdict line (learned beats all baselines / beats
    some / no signal — never a statistical-significance claim, matching phase15_comparison.py's
    headline-delta philosophy), with a RARE-TARGET caveat when the expected positive count in the
    test split (base_rate x n_test) falls under the contracts §7 20-positive caution threshold;
  - a no-aggregate-score note (D22/V2 TDD §6: every target/metric is reported separately, never
    combined into one number, matching every priorphaseN comparison/scenario script in this repo).

It also copies whichever calibration-<target>.csv files (contracts §5 frozen header:
`bin,mean_pred,mean_actual,count`) it is pointed at (`--calibration-dir`, repeatable) into
`<out>/calibration/<source-dir-slug>/` — a straight file copy (byte-for-byte); this script never
parses their contents (that is scripts/plot_results.py's plot_calibration()'s job).

TARGET-NAME AMBIGUITY (flagged, not resolved by adaptation): docs/design/P22-CONTRACTS.md §5 names
six binary targets literally (`completed, liked, shared, followed, not_interested,
session_exit(=observed_exit_after_impression)`) and one linear target literally (`watch_ratio`),
but the eighth target — the survey-satisfaction proxy regressor trained "only with --survey and
survey.csv present" — is never given an exact `target` column string anywhere in the frozen
contracts, training_log_schema.hpp, or config.hpp read for this package. This script does NOT
guess a name: every function here discovers the set of targets (and splits, and which of
auc/rmse is populated per target) directly from whatever `target` values appear in the given
`--eval` CSV(s), so it renders correctly regardless of what package B eventually names that eighth
target (or any other target-set change).

This script contains no simulation/training logic (design decision D15, matching every
phaseN_comparison.py/phase21_scenarios.py in this repo): it only reads training_eval.csv,
copies calibration CSVs verbatim, and renders `offline_eval.md`. Deliberately self-contained (no
sys.path-relative import of phase15_comparison.py, unlike phase21_scenarios.py) — this domain
(training_eval.csv rows) shares no data shape with phase15_comparison's summary.json arms, so
there is nothing genuine to reuse beyond the `fmt`/`warn`/`render_table` CONVENTIONS, which are
reproduced here in miniature instead of adding a cross-script import for a handful of one-line
helpers.

Usage
-----
    python3 scripts/phase22_report.py \\
        --eval results/phase22/models-temporal/training_eval.csv,\\
results/phase22/models-user_disjoint/training_eval.csv \\
        [--calibration-dir results/phase22/models-temporal] \\
        [--calibration-dir results/phase22/models-user_disjoint] \\
        [--out results/published/phase22] [--precision 6]

    --eval CSV[,CSV...]   one or two training_eval.csv paths (contracts §5 frozen header). Splits
                          are read from each file's own `split` column, NOT inferred from
                          argument position/order — a single combined file with both splits'
                          rows works identically to two separate per-split files.
    --calibration-dir DIR directory to copy calibration-<target>.csv files FROM (repeatable; e.g.
                          once per split's train_models --out-dir). Optional; omit to skip.
    --out DIR             output directory (default: results/published/phase22)
    --precision N         significant figures for floats (default: 6)

    python3 scripts/phase22_report.py --self-test
        Builds synthetic training_eval.csv/calibration-*.csv fixtures under a tempdir, exercises
        the loaders/verdict logic/renderer directly AND a full in-process main() call (both the
        two-CSV and single-CSV --eval forms, plus CLI-error and missing-file paths), asserts
        expected cells, prints PASS/FAIL lines, cleans up, and exits. Ignores every other argument.

    (No third-party dependencies -- plain python3 is enough, same as every phaseN_comparison.py;
    only scripts/plot_results.py's additions need the uv/3.12 environment.)

Exit status: 0 if at least one `--eval` CSV was readable and produced at least one target section
(or --self-test passed), 1 if none did, 2 on a CLI usage error (e.g. missing --eval without
--self-test).
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# contracts §5 frozen header (training_eval.csv) and model set.
EVAL_REQUIRED_COLUMNS = [
    "target", "model", "split", "n_train", "n_test", "auc", "log_loss", "rmse",
    "calibration_slope", "calibration_intercept", "base_rate",
]
MODELS = ["learned", "global_frequency", "per_source_frequency", "served_score"]
BASELINES = ["global_frequency", "per_source_frequency", "served_score"]

# contracts §7: offline tests SKIP-with-reason "targets whose base rate at reduced scale is <20
# positives" -- reused here as the caution threshold for this report's rare-target verdict note
# (expected positives in the TEST split = base_rate * n_test).
RARE_POSITIVE_THRESHOLD = 20

COMMON_COLS = [("n_train", "n_train"), ("n_test", "n_test"), ("base_rate", "base_rate")]
BINARY_METRIC_COLS = [
    ("auc", "AUC"), ("log_loss", "log-loss"),
    ("calibration_slope", "calib. slope"), ("calibration_intercept", "calib. intercept"),
]
CONTINUOUS_METRIC_COLS = [
    ("rmse", "RMSE"),
    ("calibration_slope", "calib. slope"), ("calibration_intercept", "calib. intercept"),
]
AMBIGUOUS_METRIC_COLS = [
    ("auc", "AUC"), ("log_loss", "log-loss"), ("rmse", "RMSE"),
    ("calibration_slope", "calib. slope"), ("calibration_intercept", "calib. intercept"),
]


def warn(message: str) -> None:
    print(f"phase22_report: {message}", file=sys.stderr)


def fmt(value, precision: int) -> str:
    """Same shape as phase15_comparison.py's fmt(), plus explicit NaN rendering (contracts' own
    vocabulary -- "NaN where inapplicable" -- rather than Python's lowercase 'nan')."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.{precision}g}"
    return str(value)


def render_table(header: list, rows: list) -> str:
    """Left-justified, padded, pipe-delimited markdown table (same visual style as
    phase15_comparison.py's render_markdown_table / phase21_scenarios.py's render_table)."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells: list) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(header), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


# --- Parsing --------------------------------------------------------------------------------


def _parse_float(s: Optional[str]) -> float:
    """Tolerant float parse: treats '', 'nan'/'NaN', 'null', 'none'/'None' (any case) as NaN, and
    any other unparseable string as NaN too (warned by the caller's row-level context, not here --
    a single bad cell should not sink the whole file)."""
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


@dataclass
class EvalCell:
    n_train: Optional[int]
    n_test: Optional[int]
    auc: float
    log_loss: float
    rmse: float
    calibration_slope: float
    calibration_intercept: float
    base_rate: float


def read_eval_csv(path: Path) -> list:
    """Raw (target, model, split, EvalCell) tuples from one training_eval.csv, or [] if the file
    is absent, unreadable, or missing a required frozen column -- never raises. Rows missing
    target/model/split entirely are skipped (warned); numeric parse failures degrade to NaN/None
    per-cell rather than dropping the row (contracts: "NaN where inapplicable" is itself a valid,
    expected value, not a parse error)."""
    if not path.exists():
        warn(f"--eval path not found: {path}")
        return []
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            missing = [c for c in EVAL_REQUIRED_COLUMNS if c not in fieldnames]
            if missing:
                warn(f"{path}: missing required column(s) {missing}; treating file as unusable")
                return []
            out = []
            for row in reader:
                target, model, split = row.get("target"), row.get("model"), row.get("split")
                if not target or not model or not split:
                    warn(f"{path}: skipping row with empty target/model/split: {row}")
                    continue
                cell = EvalCell(
                    n_train=_parse_int(row.get("n_train")),
                    n_test=_parse_int(row.get("n_test")),
                    auc=_parse_float(row.get("auc")),
                    log_loss=_parse_float(row.get("log_loss")),
                    rmse=_parse_float(row.get("rmse")),
                    calibration_slope=_parse_float(row.get("calibration_slope")),
                    calibration_intercept=_parse_float(row.get("calibration_intercept")),
                    base_rate=_parse_float(row.get("base_rate")),
                )
                out.append((target, model, split, cell))
            return out
    except Exception as exc:  # malformed CSV / IO error
        warn(f"failed to read {path}: {exc}")
        return []


def load_eval_data(paths: list):
    """Aggregates any number of training_eval.csv paths into data[(target, split, model)] ->
    EvalCell, plus the sorted sets of targets/splits actually present. Splits are read from each
    row's OWN `split` column (never inferred from which --eval path a row came from), so a single
    combined file with both splits' rows behaves identically to two separate per-split files. A
    (target, split, model) key repeated across files (or within one) is overwritten by the LAST
    occurrence -- an edge case not expected in practice (each train_models invocation writes its
    own --out-dir's training_eval.csv), not worth a special-cased warning.
    """
    data: dict = {}
    targets: set = set()
    splits: set = set()
    for path in paths:
        for target, model, split, cell in read_eval_csv(Path(path)):
            data[(target, split, model)] = cell
            targets.add(target)
            splits.add(split)
    return data, targets, splits


# --- Target-kind detection + column selection ------------------------------------------------


def target_kind(target: str, splits: list, data: dict) -> str:
    """'binary' (AUC populated, RMSE not), 'continuous' (RMSE populated, AUC not), 'ambiguous'
    (both populated -- unexpected, handled defensively rather than crashing) or 'unavailable'
    (neither populated for this target in any split/model given)."""
    saw_auc = saw_rmse = False
    for split in splits:
        for model in MODELS:
            cell = data.get((target, split, model))
            if cell is None:
                continue
            saw_auc = saw_auc or not math.isnan(cell.auc)
            saw_rmse = saw_rmse or not math.isnan(cell.rmse)
    if saw_auc and saw_rmse:
        return "ambiguous"
    if saw_auc:
        return "binary"
    if saw_rmse:
        return "continuous"
    return "unavailable"


def metric_cols_for_kind(kind: str) -> list:
    if kind == "binary":
        return COMMON_COLS + BINARY_METRIC_COLS
    if kind == "continuous":
        return COMMON_COLS + CONTINUOUS_METRIC_COLS
    return COMMON_COLS + AMBIGUOUS_METRIC_COLS  # 'ambiguous' or 'unavailable': show everything


# --- Rendering: per-target table + verdict ----------------------------------------------------


def render_target_table(target: str, kind: str, splits: list, data: dict, precision: int) -> str:
    cols = metric_cols_for_kind(kind)
    header = ["model"]
    for split in splits:
        header.extend(f"{split}: {label}" for _key, label in cols)

    rows = []
    for model in MODELS:
        row = [model]
        for split in splits:
            cell = data.get((target, split, model))
            for key, _label in cols:
                row.append("n/a (no row)" if cell is None else fmt(getattr(cell, key), precision))
        rows.append(row)
    return render_table(header, rows)


def split_verdict(target: str, split: str, kind: str, data: dict, precision: int) -> str:
    """One clause covering a single (target, split) pair -- see target_verdict_line() for how
    clauses across splits are joined into the single verdict LINE this package's brief asks for.
    Purely mechanical (learned's metric strictly beats / does not beat each baseline's), matching
    phase15_comparison.py's headline-delta philosophy: no statistical-significance claim, ever.
    """
    learned = data.get((target, split, "learned"))
    if learned is None:
        return f"{split}: learned model row missing -- no verdict possible"

    if kind == "binary":
        metric, higher_is_better = "auc", True
    elif kind == "continuous":
        metric, higher_is_better = "rmse", False
    else:
        return (f"{split}: target kind {kind} (AUC/RMSE both or neither populated) -- no "
                f"mechanical verdict computed")

    learned_value = getattr(learned, metric)
    if math.isnan(learned_value):
        return f"{split}: learned's {metric} unavailable -- no verdict possible"

    comparable = []
    for baseline in BASELINES:
        cell = data.get((target, split, baseline))
        if cell is None:
            continue
        value = getattr(cell, metric)
        if not math.isnan(value):
            comparable.append((baseline, value))

    if not comparable:
        verdict = f"no comparable baseline {metric} values available"
    else:
        beats = [b for b, v in comparable
                 if (learned_value > v if higher_is_better else learned_value < v)]
        n, k = len(comparable), len(beats)
        baseline_str = ", ".join(f"{b}={fmt(v, precision)}" for b, v in comparable)
        if k == n:
            verdict = (f"learned beats all {n} baseline(s) on {metric} "
                       f"(learned={fmt(learned_value, precision)}; {baseline_str})")
        elif k == 0:
            verdict = (f"no signal -- learned ({metric}={fmt(learned_value, precision)}) does "
                       f"not beat any baseline ({baseline_str})")
        else:
            verdict = (f"learned beats {k}/{n} baselines on {metric} "
                       f"(learned={fmt(learned_value, precision)}; {baseline_str})")

    rare_note = ""
    base_rate, n_test = learned.base_rate, learned.n_test
    if not math.isnan(base_rate) and n_test:
        expected_positives = base_rate * n_test
        if expected_positives < RARE_POSITIVE_THRESHOLD:
            rare_note = (
                f"; RARE TARGET -- base_rate={fmt(base_rate, precision)} x n_test={n_test} ~= "
                f"{expected_positives:.1f} expected positive(s) in the held-out test split, below "
                f"the contracts §7 {RARE_POSITIVE_THRESHOLD}-positive caution threshold: interpret "
                f"{metric} cautiously"
            )

    return f"{split}: {verdict}{rare_note}"


def target_verdict_line(target: str, kind: str, splits: list, data: dict, precision: int) -> str:
    clauses = [split_verdict(target, split, kind, data, precision) for split in splits]
    return "VERDICT: " + "; ".join(clauses)


def render_target_section(target: str, splits: list, data: dict, precision: int) -> str:
    kind = target_kind(target, splits, data)
    parts = [f"## Target: {target}\n"]
    if kind == "ambiguous":
        parts.append("*(both AUC and RMSE are populated for this target across the given data -- "
                     "unexpected; showing every metric column defensively.)*\n")
    elif kind == "unavailable":
        parts.append("*(neither AUC nor RMSE is populated for this target in the given data -- "
                     "showing every metric column defensively; likely an upstream write issue.)*\n")
    parts.append(render_target_table(target, kind, splits, data, precision))
    parts.append("")
    parts.append(target_verdict_line(target, kind, splits, data, precision))
    parts.append("")
    return "\n".join(parts)


def render_markdown(data: dict, targets: list, splits: list, precision: int) -> str:
    parts = ["# Phase 22 -- Offline Evaluation: Learned Models vs. Baselines\n"]
    parts.append(
        "V2 TDD §4.19-4.20/§4.22, Tier-5 acceptance 1 (offline half; docs/design/"
        "P22-CONTRACTS.md §5/§6): per-target held-out comparison of the in-house learned model "
        "against three baselines (global frequency, per-source frequency, served-score-as-"
        "predictor) on `training_eval.csv`. Generated by `scripts/phase22_report.py`.\n"
    )

    parts.append("## Method\n")
    parts.append(
        "**Identical-world discipline** (Tier-5 acceptance / D8/D17): every number below comes "
        "from ONE pinned log-world simulation run -- seed 42, `configs/realism-medium-retention."
        "json` + `learning_v2.training_log`/`survey` gates on, `scripts/run_phase22_logworld.sh` "
        "-- so every split/model/target comparison in this report shares the identical underlying "
        "world (no re-simulation between models or splits; only the train/test partition and the "
        "predictor differ).\n\n"
        "**Split definitions** (contracts §5, both deterministic, no shuffling/rng):\n"
        "- `temporal` -- requests ordered by timestamp; the first 80% train, the remaining 20% "
        "test.\n"
        "- `user_disjoint` -- whole USERS assigned by the pinned "
        "`hash01(userId ^ kSplitSalt) < 0.8`; every request from a train-assigned user is train, "
        "every request from a test-assigned user is test, so no user straddles the boundary "
        "(unlike `temporal`, where one user's early and late requests can land on opposite "
        "sides).\n\n"
        "Both `train_models` invocations use seed 4242 (deliberately distinct from the log-world's "
        "own seed 42 -- see `scripts/run_phase22_logworld.sh`'s header) for the `\"training-split\"` "
        "/ `\"model-init\"` streams.\n"
    )

    if not targets:
        parts.append("*(no targets found in the given --eval CSV(s).)*\n")
        return "\n".join(parts) + "\n"

    parts.append("## Per-target results\n")
    parts.append(
        "One table per target, columns grouped `SPLIT: metric` for every split present in the "
        "input data (both splits side by side when two `--eval` files are given). Rows are the "
        "four frozen models (contracts §5); columns are AUC/log-loss/calibration for binary "
        "targets or RMSE/calibration for continuous targets, auto-detected per target from "
        "which of auc/rmse is populated (this report never assumes a fixed eight-target list -- "
        "see this script's module docstring on the eighth target's unspecified name). `NaN` "
        "means the source row itself carries NaN (contracts: \"inapplicable\"); `n/a (no row)` "
        "means no row at all was found for that (target, split, model) combination.\n"
    )
    for target in targets:
        parts.append(render_target_section(target, splits, data, precision))

    parts.append("## Notes\n")
    parts.append(
        "- **No aggregate score** (D22 / V2 TDD §6): every target and every metric is reported "
        "SEPARATELY above; this script never combines AUC/log-loss/RMSE/calibration across "
        "targets or models into one number, and neither do `scripts/plot_results.py`'s Phase 22 "
        "additions (`plot_offline_auc`/`plot_calibration`, one figure axis per target).\n"
        "- **Verdicts are mechanical, not statistical** (phase15_comparison.py precedent): "
        "\"beats\"/\"no signal\" above means learned's point-estimate metric is strictly better/"
        "not-better than a given baseline's point estimate on this ONE held-out split of this ONE "
        "log world -- no significance test, confidence interval, or claim about a DIFFERENT log "
        "world is made anywhere in this report.\n"
        "- **Rare-target caveat**: a target flagged `RARE TARGET` has fewer than the contracts §7 "
        "20-positive caution threshold's worth of expected positives (`base_rate x n_test`) in "
        "the held-out test split -- AUC on that few positives is high-variance; read the verdict "
        "as directional, not conclusive.\n"
        "- **calibration-<target>.csv** (contracts §5: `bin,mean_pred,mean_actual,count`, 10 "
        "equal-count bins) is copied verbatim into `<out>/calibration/<source-dir-slug>/` when "
        "`--calibration-dir` is given -- see `scripts/plot_results.py`'s `plot_calibration()` for "
        "the reliability-diagram rendering of those files (this script does not parse them).\n"
    )
    return "\n".join(parts) + "\n"


# --- Calibration CSV copy ----------------------------------------------------------------------


def _slug(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return s or "dir"


def copy_calibration_csvs(dirs: list, out: Path) -> int:
    """Copies every calibration-*.csv (contracts §5 frozen filename pattern) found directly under
    each given directory into `<out>/calibration/<slug(dir.name)>/`, byte-for-byte (this script
    never parses their contents). Returns the total file count copied. Warns (never raises) on a
    non-directory path or a directory with zero matching files."""
    copied = 0
    for raw in dirs:
        d = Path(raw)
        if not d.is_dir():
            warn(f"--calibration-dir {d}: not a directory; skipping")
            continue
        matches = sorted(d.glob("calibration-*.csv"))
        if not matches:
            warn(f"--calibration-dir {d}: no calibration-*.csv files found; skipping")
            continue
        dest_dir = out / "calibration" / _slug(d.name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in matches:
            shutil.copy2(src, dest_dir / src.name)
            copied += 1
    return copied


# --- CLI -----------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Phase 22 offline-evaluation report (docs/design/P22-CONTRACTS.md "
                    "§6) from training_eval.csv as offline_eval.md, and copy calibration CSVs.",
    )
    parser.add_argument("--eval", default=None,
                        help="one or two training_eval.csv paths, comma-separated (contracts §5 "
                             "frozen header). Required unless --self-test.")
    parser.add_argument("--calibration-dir", action="append", dest="calibration_dir",
                        default=None, metavar="DIR",
                        help="directory to copy calibration-<target>.csv files from into "
                             "<out>/calibration/<dir-slug>/; repeatable (e.g. once per split). "
                             "Optional; omit to skip calibration copying.")
    parser.add_argument("--out", type=Path, default=Path("results/published/phase22"),
                        help="output directory for offline_eval.md + calibration/ "
                             "(default: results/published/phase22)")
    parser.add_argument("--precision", type=int, default=6,
                        help="significant figures for floats (default: 6)")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="run an in-script synthetic-fixture smoke test end-to-end, print "
                             "PASS/FAIL lines, and exit (ignores every other argument; no "
                             "committed fixture files)")
    args = parser.parse_args(argv)
    if not args.self_test and not args.eval:
        parser.error("--eval is required (unless --self-test)")
    return args


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    if args.self_test:
        return run_self_test()

    eval_paths = [p.strip() for p in args.eval.split(",") if p.strip()]
    if not eval_paths:
        warn("--eval produced zero usable paths")
        return 1

    data, targets, splits = load_eval_data(eval_paths)
    if not targets:
        warn("no usable rows found in any --eval CSV")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    copied = 0
    if args.calibration_dir:
        copied = copy_calibration_csvs(args.calibration_dir, args.out)

    md_path = args.out / "offline_eval.md"
    md_path.write_text(render_markdown(data, sorted(targets), sorted(splits), args.precision))

    print(f"phase22_report: wrote {md_path}")
    if copied:
        print(f"phase22_report: copied {copied} calibration CSV(s) into {args.out / 'calibration'}")
    return 0


# --- Self-test -------------------------------------------------------------------------------


def _check(cond: bool, message: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {message}")
    return cond


def _write_eval_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EVAL_REQUIRED_COLUMNS)
        for row in rows:
            w.writerow(row)


def _write_calibration_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bin", "mean_pred", "mean_actual", "count"])
        for b in range(10):
            w.writerow([b, 0.05 + 0.1 * b, 0.06 + 0.1 * b, 500])


def run_self_test() -> int:
    """Synthetic-fixture smoke test (no committed fixture files). Builds two training_eval.csv
    fixtures (temporal/user_disjoint) covering: a binary target learned beats all baselines on
    both splits ("completed"); a binary target with split-dependent verdicts, beats-some on
    temporal and beats-all on user_disjoint ("liked"); a RARE binary target with no signal
    ("shared"); a continuous target beating all baselines ("watch_ratio"); and a target present
    ONLY in temporal with neither AUC nor RMSE populated ("weird_target", exercises the
    'unavailable'-kind defensive path + the cross-split "n/a (no row)" path). Exercises the
    loaders/verdict logic/renderer directly AND a full in-process main() call (two-CSV form,
    single-CSV form, a missing-file warn-and-continue path, a total-failure path, and the CLI
    usage-error path). Prints one PASS/FAIL line per check."""
    print("phase22_report --self-test")
    ok = True

    nan = ""  # empty cell -> NaN per _parse_float

    with tempfile.TemporaryDirectory(prefix="phase22-report-selftest-") as tmp:
        root = Path(tmp)

        # target,model,split,n_train,n_test,auc,log_loss,rmse,calibration_slope,
        # calibration_intercept,base_rate
        temporal_rows = [
            ("completed", "learned", "temporal", 8000, 2000, 0.75, 0.55, nan, 0.95, 0.02, 0.30),
            ("completed", "global_frequency", "temporal", 8000, 2000, 0.50, 0.69, nan, 0.10, 0.28, 0.30),
            ("completed", "per_source_frequency", "temporal", 8000, 2000, 0.52, 0.68, nan, 0.20, 0.25, 0.30),
            ("completed", "served_score", "temporal", 8000, 2000, 0.55, 0.67, nan, 0.30, 0.20, 0.30),

            ("liked", "learned", "temporal", 8000, 2000, 0.60, 0.62, nan, 0.80, 0.05, 0.10),
            ("liked", "global_frequency", "temporal", 8000, 2000, 0.50, 0.65, nan, 0.10, 0.10, 0.10),
            ("liked", "per_source_frequency", "temporal", 8000, 2000, 0.55, 0.64, nan, 0.15, 0.09, 0.10),
            ("liked", "served_score", "temporal", 8000, 2000, 0.65, 0.60, nan, 0.40, 0.06, 0.10),

            ("shared", "learned", "temporal", 8000, 2000, 0.50, 0.70, nan, 0.50, 0.01, 0.003),
            ("shared", "global_frequency", "temporal", 8000, 2000, 0.55, 0.68, nan, 0.10, 0.01, 0.003),
            ("shared", "per_source_frequency", "temporal", 8000, 2000, 0.60, 0.66, nan, 0.10, 0.01, 0.003),
            ("shared", "served_score", "temporal", 8000, 2000, 0.65, 0.64, nan, 0.10, 0.01, 0.003),

            ("watch_ratio", "learned", "temporal", 8000, 2000, nan, nan, 0.10, 0.90, 0.03, nan),
            ("watch_ratio", "global_frequency", "temporal", 8000, 2000, nan, nan, 0.20, 0.10, 0.10, nan),
            ("watch_ratio", "per_source_frequency", "temporal", 8000, 2000, nan, nan, 0.25, 0.10, 0.10, nan),
            ("watch_ratio", "served_score", "temporal", 8000, 2000, nan, nan, 0.30, 0.10, 0.10, nan),

            ("weird_target", "learned", "temporal", 8000, 2000, nan, nan, nan, nan, nan, nan),
            ("weird_target", "global_frequency", "temporal", 8000, 2000, nan, nan, nan, nan, nan, nan),
            ("weird_target", "per_source_frequency", "temporal", 8000, 2000, nan, nan, nan, nan, nan, nan),
            ("weird_target", "served_score", "temporal", 8000, 2000, nan, nan, nan, nan, nan, nan),
        ]
        user_disjoint_rows = [
            ("completed", "learned", "user_disjoint", 7800, 2200, 0.72, 0.58, nan, 0.90, 0.03, 0.29),
            ("completed", "global_frequency", "user_disjoint", 7800, 2200, 0.50, 0.69, nan, 0.10, 0.28, 0.29),
            ("completed", "per_source_frequency", "user_disjoint", 7800, 2200, 0.51, 0.68, nan, 0.20, 0.25, 0.29),
            ("completed", "served_score", "user_disjoint", 7800, 2200, 0.53, 0.67, nan, 0.30, 0.20, 0.29),

            ("liked", "learned", "user_disjoint", 7800, 2200, 0.58, 0.63, nan, 0.75, 0.05, 0.11),
            ("liked", "global_frequency", "user_disjoint", 7800, 2200, 0.40, 0.66, nan, 0.10, 0.10, 0.11),
            ("liked", "per_source_frequency", "user_disjoint", 7800, 2200, 0.45, 0.65, nan, 0.15, 0.09, 0.11),
            ("liked", "served_score", "user_disjoint", 7800, 2200, 0.50, 0.64, nan, 0.20, 0.08, 0.11),

            ("shared", "learned", "user_disjoint", 7800, 2200, 0.50, 0.70, nan, 0.50, 0.01, 0.0025),
            ("shared", "global_frequency", "user_disjoint", 7800, 2200, 0.52, 0.69, nan, 0.10, 0.01, 0.0025),
            ("shared", "per_source_frequency", "user_disjoint", 7800, 2200, 0.53, 0.68, nan, 0.10, 0.01, 0.0025),
            ("shared", "served_score", "user_disjoint", 7800, 2200, 0.54, 0.67, nan, 0.10, 0.01, 0.0025),

            ("watch_ratio", "learned", "user_disjoint", 7800, 2200, nan, nan, 0.11, 0.88, 0.03, nan),
            ("watch_ratio", "global_frequency", "user_disjoint", 7800, 2200, nan, nan, 0.21, 0.10, 0.10, nan),
            ("watch_ratio", "per_source_frequency", "user_disjoint", 7800, 2200, nan, nan, 0.24, 0.10, 0.10, nan),
            ("watch_ratio", "served_score", "user_disjoint", 7800, 2200, nan, nan, 0.29, 0.10, 0.10, nan),
        ]

        temporal_csv = root / "models-temporal" / "training_eval.csv"
        user_disjoint_csv = root / "models-user_disjoint" / "training_eval.csv"
        _write_eval_csv(temporal_csv, temporal_rows)
        _write_eval_csv(user_disjoint_csv, user_disjoint_rows)

        calib_dir_t = root / "models-temporal"
        calib_dir_u = root / "models-user_disjoint"
        _write_calibration_csv(calib_dir_t / "calibration-completed.csv")
        _write_calibration_csv(calib_dir_t / "calibration-watch_ratio.csv")
        _write_calibration_csv(calib_dir_u / "calibration-completed.csv")

        # --- direct loader/logic checks ---------------------------------------------------
        data, targets, splits = load_eval_data([temporal_csv, user_disjoint_csv])
        ok &= _check(targets == {"completed", "liked", "shared", "watch_ratio", "weird_target"},
                    f"load_eval_data: discovers all five targets (got {sorted(targets)})")
        ok &= _check(splits == {"temporal", "user_disjoint"},
                    f"load_eval_data: discovers both splits (got {sorted(splits)})")

        ok &= _check(target_kind("completed", sorted(splits), data) == "binary",
                    "target_kind: 'completed' detected as binary")
        ok &= _check(target_kind("watch_ratio", sorted(splits), data) == "continuous",
                    "target_kind: 'watch_ratio' detected as continuous")
        ok &= _check(target_kind("weird_target", sorted(splits), data) == "unavailable",
                    "target_kind: all-NaN 'weird_target' detected as unavailable")

        v_completed = split_verdict("completed", "temporal", "binary", data, 6)
        ok &= _check(v_completed.startswith("temporal: learned beats all 3 baseline(s)"),
                    f"split_verdict: 'completed'/temporal is beats-all (got {v_completed!r})")

        v_liked_temporal = split_verdict("liked", "temporal", "binary", data, 6)
        ok &= _check("beats 2/3 baselines" in v_liked_temporal,
                    f"split_verdict: 'liked'/temporal is beats-2/3 (got {v_liked_temporal!r})")
        v_liked_user = split_verdict("liked", "user_disjoint", "binary", data, 6)
        ok &= _check("beats all 3 baseline(s)" in v_liked_user,
                    f"split_verdict: 'liked'/user_disjoint is beats-all, differing from its own "
                    f"temporal clause (got {v_liked_user!r})")

        v_shared = split_verdict("shared", "temporal", "binary", data, 6)
        ok &= _check("no signal" in v_shared, f"split_verdict: 'shared' is no-signal (got {v_shared!r})")
        ok &= _check("RARE TARGET" in v_shared,
                    f"split_verdict: 'shared' (base_rate=0.003) carries the rare-target note "
                    f"(got {v_shared!r})")

        v_watch = split_verdict("watch_ratio", "temporal", "continuous", data, 6)
        ok &= _check("beats all 3 baseline(s) on rmse" in v_watch,
                    f"split_verdict: 'watch_ratio' (continuous, lower-is-better) is beats-all "
                    f"(got {v_watch!r})")

        line = target_verdict_line("weird_target", "unavailable", sorted(splits), data, 6)
        ok &= _check("temporal:" in line and "user_disjoint:" in line,
                    f"target_verdict_line: joins one clause per split (got {line!r})")
        ok &= _check("no mechanical verdict computed" in line,
                    "target_verdict_line: 'unavailable' kind renders as no-verdict, not a crash")

        table = render_target_table("completed", "binary", sorted(splits), data, 4)
        ok &= _check("temporal: AUC" in table and "user_disjoint: AUC" in table,
                    "render_target_table: both splits appear side-by-side as column groups")
        ok &= _check("n/a (no row)" not in table,
                    "render_target_table: 'completed' has a row in every model/split combo")
        table_weird = render_target_table("weird_target", "unavailable", sorted(splits), data, 4)
        # 'unavailable' kind shows all 8 AMBIGUOUS_METRIC_COLS-equivalent columns (3 common + 5
        # metric); user_disjoint has zero rows for this target across all 4 models -> 4 x 8 = 32.
        ok &= _check(table_weird.count("n/a (no row)") == 32,
                    f"render_target_table: 'weird_target' (temporal-only) shows 4 models x 8 "
                    f"columns = 32 'n/a (no row)' cells for its absent user_disjoint rows (got "
                    f"{table_weird.count('n/a (no row)')})")
        ok &= _check("NaN" in table_weird,
                    "render_target_table: 'weird_target's temporal row (present but all-NaN) "
                    "renders as NaN, not 'n/a (no row)'")

        # --- full main() round trips -------------------------------------------------------
        out_dir = root / "out"
        rc = main([
            "--eval", f"{temporal_csv},{user_disjoint_csv}",
            "--calibration-dir", str(calib_dir_t), "--calibration-dir", str(calib_dir_u),
            "--out", str(out_dir), "--precision", "4",
        ])
        ok &= _check(rc == 0, f"main(): two-CSV form exits 0 (got {rc})")
        md_path = out_dir / "offline_eval.md"
        ok &= _check(md_path.exists(), "main(): writes offline_eval.md")
        md_text = md_path.read_text() if md_path.exists() else ""
        ok &= _check("## Method" in md_text, "render_markdown: Method section present")
        ok &= _check("No aggregate score" in md_text,
                    "render_markdown: no-aggregate-score note present")
        ok &= _check(all(f"## Target: {t}" in md_text for t in targets),
                    "render_markdown: every discovered target has its own section")
        ok &= _check("RARE TARGET" in md_text, "render_markdown: rare-target note surfaces in the report")

        slug_t, slug_u = _slug(calib_dir_t.name), _slug(calib_dir_u.name)
        copied_t = out_dir / "calibration" / slug_t / "calibration-completed.csv"
        copied_u = out_dir / "calibration" / slug_u / "calibration-completed.csv"
        ok &= _check(copied_t.exists() and copied_u.exists(),
                    "copy_calibration_csvs: same-named files from two --calibration-dir "
                    "arguments both land, namespaced by source-dir slug (no collision)")
        ok &= _check((out_dir / "calibration" / slug_t / "calibration-watch_ratio.csv").exists(),
                    "copy_calibration_csvs: a second file from the same dir also copies")

        # Single-CSV --eval form (only temporal): user_disjoint column group should not even be
        # attempted -- render with only one split.
        out_dir_single = root / "out-single"
        rc_single = main(["--eval", str(temporal_csv), "--out", str(out_dir_single)])
        ok &= _check(rc_single == 0, f"main(): single-CSV form exits 0 (got {rc_single})")
        md_single = (out_dir_single / "offline_eval.md").read_text()
        ok &= _check("user_disjoint:" not in md_single,
                    "main(): single-CSV form renders temporal only, no user_disjoint column group")

        # Missing-file-but-one-usable path.
        out_dir_partial = root / "out-partial"
        rc_partial = main([
            "--eval", f"{root / 'does-not-exist.csv'},{temporal_csv}",
            "--out", str(out_dir_partial),
        ])
        ok &= _check(rc_partial == 0,
                    f"main(): one missing + one usable --eval path still exits 0 (got {rc_partial})")

        # Total-failure path.
        rc_fail = main(["--eval", str(root / "does-not-exist.csv"), "--out", str(root / "out-fail")])
        ok &= _check(rc_fail == 1, f"main(): every --eval path unusable exits 1 (got {rc_fail})")

        # CLI usage-error path.
        rc_usage = main(["--out", str(root / "out-usage")])
        ok &= _check(rc_usage == 2, f"main(): missing --eval (no --self-test) exits 2 (got {rc_usage})")

    print(f"phase22_report --self-test: {'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
