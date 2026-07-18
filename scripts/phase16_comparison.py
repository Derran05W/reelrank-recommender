#!/usr/bin/env python3
"""phase16_comparison.py — Phase 16 four-arm session-dynamics comparison (V2 TDD §4.6-4.9, §6).

Extends `phase15_comparison.py` (imported directly for its purely-mechanical reading/formatting
helpers and the unchanged P15 metric definitions -- design decision D15: this script contains no
simulation logic, only reads result files and renders tables) with the Phase 16 SESSION HEALTH
metric group (V2 TDD §6 "session health"): session length, exit-type shares, early-failure-exit
rate, session utility (U_s), regret/satisfaction per minute, and next-session starting
satisfaction (V2 TDD §4.9, plan Phase 16 task 5).

Given the four Phase 16 arm result directories (semantic / engagement-optimized /
satisfaction-proxy / oracle_satisfaction, as produced by scripts/run_phase16_experiment.sh on the
"-sessions" config trio -- realism.session_dynamics on, in addition to Phase 15's content_v2 +
latent_reactions), this reads each arm's summary.json (+ config.json for the arm-identity section,
+ best-effort reads of welfare_archetype_metrics.csv for per-archetype exposure) and writes:

  comparison.csv   one row per arm, one column per §4.4 report-list metric (UNCHANGED from Phase
                   15 -- the engagement/welfare/recommendation-quality groups).
  comparison.md    headline deltas (P15's + the new Phase 16 session-health headline), an
                   arm-identity section, the P15 per-arm table, the NEW session-health panel
                   (mean session length, exit-type shares, early-failure-exit rate, U_s, regret/
                   satisfaction per minute, next-session starting satisfaction), the per-archetype
                   exposure table, and notes.

Arm order is always semantic, engagement, proxy, oracle (deterministic ordering, regardless of CLI
argument order or which arms are available) -- a missing/unreadable arm still gets a row, with every
cell reported as unavailable rather than the row being silently dropped, so the output shape never
changes across invocations. Same contract as phase15_comparison.py.

Package A/B dependency (read defensively, mirroring phase15_comparison.py's package-A/B2 notes):
  Phase 16 is developed as parallel packages (plan Phase 16 "Suggested package split"): package A
  owns HiddenSessionState + fatigue dynamics, package B owns the exit model + classification +
  session-health metrics/U_s -- both in worktrees separate from the one this script ships from
  (package C: experiment tooling). Until their work is merged, no run in this worktree ever closes
  a real session (Simulator::stepV2's `closedSession` out-param is an unwired stub -- see
  tests/property/session_exit_statistical_test.cpp's header comment for the full contract this
  script and that test are BOTH written against), so:
    - summary.json carries no top-level "session_health" block at all yet.
  Every session-health cell below is read through `_session_health_metric`, which tries a small
  list of plausible candidate key names per metric (the same defensive-multi-candidate idiom
  phase15_comparison.py already uses for satisfaction_per_minute) and renders
  "n/a (pre-integration)" when none are present -- no exception, no crash, and the moment package
  B lands a `session_health` block under any of the candidate names, these cells populate with NO
  changes needed to this script.
  Unlike Phase 15's oracle (pending package B2 there), oracle_satisfaction is a Phase-15
  deliverable and is expected to already work in a Phase-16 worktree; this script no longer
  special-cases it as "pending" -- a missing oracle summary.json here just means the arm wasn't
  run / wasn't passed on the command line.

Usage
-----
    python3 scripts/phase16_comparison.py \\
        --semantic <dir> --engagement <dir> --proxy <dir> [--oracle <dir>] \\
        [--out results/published/phase16] [--precision 6]

    (No third-party dependencies -- plain python3 is enough; `uv run --project scripts` also works
    since scripts/ is a valid uv project, but is not required here.)

Exit status: 0 if at least one of the three required arms (semantic/engagement/proxy) had a
readable summary.json, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling-module import work regardless of CWD (scripts/ is not a package -- both scripts are
# plain, independently-runnable files per D15/TDD §22, so this is a straight file-relative import,
# not a package import).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase15_comparison as p15  # noqa: E402  (see sys.path.insert above)

ARM_ORDER = p15.ARM_ORDER  # ["semantic", "engagement", "proxy", "oracle"] -- unchanged ordering

ARM_DESCRIPTION = {
    "semantic": "semantic-similarity ranker (algorithm: hnsw, configs/realism-medium-sessions.json)",
    "engagement": "engagement-optimized preset (algorithm: hnsw_ranker, "
                  "configs/realism-medium-sessions-engagement.json)",
    "proxy": "satisfaction-proxy preset (algorithm: hnsw_ranker, "
             "configs/realism-medium-sessions-proxy.json)",
    "oracle": "oracle_satisfaction, EVALUATION-ONLY upper bound "
              "(configs/realism-medium-sessions.json)",
}

NA = p15.NA_PRE_INTEGRATION  # "n/a (pre-integration)" -- reused verbatim (same meaning here)

# --- Session-health metric group (V2 TDD §6/§4.9, plan Phase 16 task 5) -------------------------
# One entry per reported scalar: (column key, header, candidate summary.json key names under the
# "session_health" block). Multiple candidates per metric because package B's exact field names
# are not fixed yet from this worktree (parallel package, see module docstring); the FIRST present
# numeric candidate wins. This list is this script's documented, best-effort CONTRACT for the
# session_health block's shape -- if package B lands different names entirely, extend the
# candidate tuples below (the rest of the script is written only against this table).
SESSION_HEALTH_METRICS: list[tuple[str, str, tuple[str, ...]]] = [
    ("mean_duration_seconds", "mean session duration (time-before-exit, s)",
     ("mean_duration_seconds", "mean_session_duration_seconds", "mean_time_before_exit_seconds",
      "time_before_exit_seconds")),
    ("mean_impressions_per_session", "mean impressions/session",
     ("mean_impressions_per_session", "mean_session_impressions")),
    ("early_failure_exit_rate", "early-failure-exit rate",
     ("early_failure_exit_rate", "failure_exit_rate")),
    ("natural_completion_rate", "natural-completion (satisfied-exit) rate",
     ("natural_completion_rate", "satisfied_exit_rate")),
    ("mean_session_utility", "U_s mean (session utility)",
     ("mean_session_utility", "mean_u_s", "u_s_mean", "session_utility_mean")),
    ("regret_per_minute", "regret per minute",
     ("regret_per_minute", "mean_regret_per_minute")),
    ("satisfaction_per_minute", "satisfaction per minute (session-scoped)",
     ("satisfaction_per_minute", "mean_satisfaction_per_minute")),
    ("next_session_starting_satisfaction", "next-session starting satisfaction",
     ("next_session_starting_satisfaction", "mean_starting_satisfaction",
      "starting_satisfaction_mean")),
    ("return_probability", "probability of returning",
     ("return_probability", "probability_of_returning", "return_rate")),
]

# Exit-type taxonomy (V2 TDD §4.8; include/rr/simulation/hidden/hidden_session_state.hpp's
# SessionExitType) plus "open" for still-open-at-run-end sessions (RunEnded -- excluded from
# exit-rate denominators per that header's comment, reported here as its own share for visibility).
# (share column key suffix, label, candidate key names)
EXIT_TYPE_SHARES: list[tuple[str, str, tuple[str, ...]]] = [
    ("failure", "failure", ("failure_share", "failure_exit_share")),
    ("satisfied", "satisfied", ("satisfied_share", "satisfied_exit_share")),
    ("fatigue", "fatigue", ("fatigue_share", "fatigue_exit_share")),
    ("external", "external", ("external_share", "external_exit_share")),
    ("regret", "regret", ("regret_share", "regret_exit_share")),
    ("open", "open (still-open at run end / RunEnded)",
     ("open_share", "still_open_share", "run_ended_share")),
]


class Phase16Arm(p15.Arm):
    """p15.Arm, with the Phase-16-appropriate "unavailable" reason (oracle is no longer pending
    package B2 -- it is a Phase 15 deliverable expected to already work) and a session-health
    reader on top of the inherited summary/config/archetype-exposure loading."""

    def _na(self) -> str:
        return NA

    def session_health_metric(self, candidates: tuple[str, ...]):
        if not self.available:
            return NA
        value = _first_present(p15._get(self.summary, "session_health"), candidates)
        return NA if value is None else value

    def exit_share(self, candidates: tuple[str, ...]):
        return self.session_health_metric(candidates)


def _first_present(section, candidates: tuple[str, ...]):
    """First candidate key in `section` (a dict, typically summary["session_health"]) holding a
    numeric value; None if `section` is absent or none of the candidates are present/numeric."""
    if not isinstance(section, dict):
        return None
    for key in candidates:
        value = section.get(key)
        if p15._is_num(value):
            return value
    return None


def load_arms(args: argparse.Namespace) -> dict[str, Phase16Arm]:
    dirs = {
        "semantic": args.semantic,
        "engagement": args.engagement,
        "proxy": args.proxy,
        "oracle": args.oracle,
    }
    return {name: Phase16Arm(name, d) for name, d in dirs.items()}


def render_arms_section(arms: dict[str, Phase16Arm]) -> str:
    lines = []
    for name in ARM_ORDER:
        arm = arms[name]
        lines.append(f"- **{name}** -- {ARM_DESCRIPTION[name]}")
        if not arm.available:
            lines.append("  - status: NOT AVAILABLE (no readable summary.json)")
            continue
        weights = arm.distinguishing_weights()
        weights_str = ", ".join(f"{k}={v:g}" for k, v in weights.items()) if weights else "(all zero -- V1 parity)"
        lines.append(f"  - directory: `{arm.directory}`")
        lines.append(f"  - non-zero V2 ranking weights: {weights_str}")
    return "\n".join(lines)


def render_session_health_table(arms: dict[str, Phase16Arm], precision: int) -> str:
    header = ["arm"] + [h for _key, h, _cands in SESSION_HEALTH_METRICS]
    rows = []
    for name in ARM_ORDER:
        arm = arms[name]
        row = [name]
        for _key, _header, candidates in SESSION_HEALTH_METRICS:
            row.append(p15.fmt(arm.session_health_metric(candidates), precision))
        rows.append(row)
    return _render_table(header, rows)


def render_exit_share_table(arms: dict[str, Phase16Arm], precision: int) -> str:
    header = ["exit type"] + ARM_ORDER
    rows = []
    for _key, label, candidates in EXIT_TYPE_SHARES:
        row = [label]
        for name in ARM_ORDER:
            row.append(p15.fmt(arms[name].exit_share(candidates), precision))
        rows.append(row)
    return _render_table(header, rows)


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(header), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def build_session_health_headline(arms: dict[str, Phase16Arm], precision: int) -> list[str]:
    """The Phase 16 headline (plan task 5's mandated framing + data-driven deltas on the new
    session-health group). Mirrors phase15_comparison.build_headline's "not available" fallback
    when either side of a comparison is not a number -- expected pre-integration in this worktree.
    """
    lines = [
        "**U_s and regret/min separate the engagement arm from the proxy arm at comparable "
        "durations.** Raw session length (time-before-exit) is deliberately NOT the headline "
        "metric here: V2 TDD §4.9 is explicit that \"a four-hour session should not automatically "
        "be considered better than a focused twenty-minute session\" -- a long session padded with "
        "regret-laden ragebait and a short, genuinely satisfying session can both show up as "
        "\"engaged\" time-on-app, and only the duration-NORMALIZED measures (satisfaction per "
        "minute, regret per minute) plus the exit-composition breakdown (satisfied vs regret/"
        "failure share) tell them apart. Session utility U_s itself already bakes in this "
        "penalty (U_s = Σsatisfaction − λ1·Σregret − λ2·harmfulFatigue − λ3·earlyFailureExit, "
        "V2 TDD §4.9), so an arm can look competitive on raw duration while trailing badly on U_s "
        "the moment its sessions run comparably long to a healthier arm's.",
    ]

    def delta_line(label: str, candidates: tuple[str, ...], name_a: str, name_b: str) -> str:
        a, b = arms[name_a], arms[name_b]
        va = _first_present(p15._get(a.summary, "session_health"), candidates) if a.available else None
        vb = _first_present(p15._get(b.summary, "session_health"), candidates) if b.available else None
        if not (p15._is_num(va) and p15._is_num(vb)):
            return f"- {label}: not available (needs a `session_health` summary.json block for both arms)."
        delta = ((va - vb) / abs(vb) * 100.0) if vb != 0 else None
        delta_str = f"{delta:+.1f}%" if delta is not None else "n/a (baseline is 0)"
        return f"- {label}: **{p15.fmt(va, precision)}** vs **{p15.fmt(vb, precision)}** ({delta_str})."

    lines.append(delta_line("U_s mean: engagement vs proxy",
                             ("mean_session_utility", "mean_u_s", "u_s_mean", "session_utility_mean"),
                             "engagement", "proxy"))
    lines.append(delta_line("regret per minute: engagement vs proxy",
                             ("regret_per_minute", "mean_regret_per_minute"), "engagement", "proxy"))
    lines.append(delta_line("mean session duration: engagement vs proxy",
                             ("mean_duration_seconds", "mean_session_duration_seconds",
                              "mean_time_before_exit_seconds", "time_before_exit_seconds"),
                             "engagement", "proxy"))
    lines.append(delta_line("early-failure-exit rate: semantic (bottom-quality reference) vs proxy",
                             ("early_failure_exit_rate", "failure_exit_rate"), "semantic", "proxy"))
    return lines


def render_markdown(arms: dict[str, Phase16Arm], rows: list[list[str]], precision: int) -> str:
    parts = []
    parts.append("# Phase 16 -- Session Dynamics: Fatigue, Probabilistic Exit, Session Outcomes\n")
    parts.append(
        "V2 TDD §4.6-4.9 session-dynamics re-run of the §4.4 core experiment: semantic-similarity, "
        "engagement-optimized, satisfaction-proxy, and (evaluation-only) oracle_satisfaction arms "
        "on `configs/realism-medium-sessions.json` (gates `content_v2`/`latent_reactions`/"
        "`session_dynamics` on) and its two weight-preset variants, seed 42. Generated by "
        "`scripts/phase16_comparison.py`. The engagement/welfare/recommendation-quality groups and "
        "per-archetype exposure below are UNCHANGED in definition from "
        "`results/published/phase15/comparison.md`; this report ADDS the session-health group "
        "(V2 TDD §6) that Phase 15 could not report.\n"
    )

    parts.append("## Headline\n")
    parts.extend(build_session_health_headline(arms, precision))
    parts.append("")
    parts.append("Engagement-vs-satisfaction headline (unchanged definitions from Phase 15):\n")
    parts.extend(p15.build_headline(arms, precision))
    parts.append(
        "\nDeltas above are computed directly from each arm's summary.json; they make no claim of "
        "statistical significance by themselves (see Notes) -- read them alongside the full "
        "per-arm table below.\n"
    )

    parts.append("## Arms\n")
    parts.append(render_arms_section(arms))
    parts.append("")

    parts.append("## Session health (V2 TDD §6, new in Phase 16)\n")
    parts.append(
        "Time-before-exit, impressions/session, early-failure-exit rate, natural-completion "
        "(satisfied-exit) rate, session utility (U_s), regret/satisfaction per minute, and "
        "next-session starting satisfaction (plan Phase 16 task 5). See the Headline section above "
        "for why raw duration is deliberately not the star of this table.\n"
    )
    parts.append(render_session_health_table(arms, precision))
    parts.append("")
    parts.append(
        "Exit-type shares (of closed sessions; \"open\" is still-open-at-run-end sessions, "
        "reported separately and excluded from the other exit-rate denominators per "
        "`SessionExitType::RunEnded`'s documented convention):\n"
    )
    parts.append(render_exit_share_table(arms, precision))
    parts.append("")

    parts.append("## Per-arm results (engagement / hidden welfare / recommendation quality)\n")
    parts.append(
        "See `comparison.csv` for the same table in machine-readable form. Columns follow the V2 "
        "TDD §4.4 report list: engagement (watch time, completion/like/share/follow, "
        "comment/save/profile-visit), hidden welfare (mean immediate satisfaction, regret, "
        "satisfaction/minute), and recommendation-quality basics (mean true affinity, "
        "estimated<->hidden cosine). Unchanged in definition from Phase 15.\n"
    )
    parts.append(p15.render_markdown_table(rows))
    parts.append("")

    parts.append("## Per-archetype exposure\n")
    parts.append(
        "Does the engagement arm over-serve ragebait/clickbait relative to the semantic baseline "
        "and the satisfaction-proxy arm? (V2 TDD §4.4 archetype catalog: genuinely-satisfying, "
        "useful, ragebait, clickbait, comfort, polished-irrelevant, niche-treasure, "
        "background-music.) Shares should sum to ~1.0 per arm column. Unchanged in definition from "
        "Phase 15 -- reused here since session dynamics does not change exposure measurement.\n"
    )
    parts.append(p15.render_archetype_section(arms, precision))
    parts.append("")

    parts.append("## Notes\n")
    parts.append(
        "- **Session health is the new group this phase reports** (V2 TDD §6): session exits "
        "(classified failure/satisfied/fatigue/external/regret), session duration, session "
        "utility (U_s), and the per-minute welfare rates. This worktree is Package C "
        "(experiment tooling) of Phase 16; packages A (`HiddenSessionState` + fatigue dynamics) "
        "and B (exit model + classification + session-health metrics/U_s) are parallel worktrees "
        "not yet merged here, so every session-health cell above reads `n/a (pre-integration)` "
        "until that merge lands a `session_health` summary.json block -- see this script's module "
        "docstring and `tests/property/session_exit_statistical_test.cpp`'s header comment for the "
        "exact contract (candidate key names) both are written against. No edits to this script "
        "are expected to be needed at that point, only re-running it against real output.\n"
        "- **Concurrency-contention caveat**: if the arms were run CONCURRENTLY "
        "(`scripts/run_phase16_experiment.sh`'s default mode), this run's wall-clock/timing "
        "numbers carry cache and memory-bandwidth contention (same caveat as Phase 15's four "
        "concurrent arms). Every other number in this report is deterministic (rng/clock-free, "
        "D8/D9) and unaffected by contention -- cross-arm comparisons are like-for-like regardless "
        "of run mode. Phase 16 arms are EXPECTED to run FASTER than the Phase 15 numbers "
        "(semantic ~124.5s, oracle ~145.8s, engagement ~556.0s, proxy ~557.7s, all Release/"
        "concurrent -- see `results/published/phase15/*/summary.json`'s "
        "`timing.total_wall_seconds`) because probabilistic exits truncate a user's remaining "
        "feed consumption once fired, doing less total per-user work than the fixed "
        "`interactions_per_user` budget implies; see `scripts/run_phase16_experiment.sh`'s header "
        "for the full reasoning. This is a qualitative expectation for the integrator to confirm, "
        "not a number re-derived here.\n"
        "- **\"Regret\" here is HIDDEN welfare regret**, not recommendation regret: the per-arm "
        "table's `hidden regret` column is `welfare.mean_regret` (LatentReaction.regret, \"wishes "
        "they had skipped\", V2 TDD §4.3, D18 evaluation carve-out) -- a different quantity from "
        "each arm's own `oracle.mean_regret` (a true-affinity gap vs. the oracle's exhaustive "
        "top-k, the V1 Phase-4 recommendation-quality regret) and from the session-health group's "
        "`regret_per_minute` above (Σregret_t over a SESSION divided by session watch-minutes, "
        "V2 TDD §4.9) -- three distinct \"regret\" quantities in this report, each documented at "
        "first use.\n"
        "- **`satisfaction_per_minute`** in the per-arm table (hidden-welfare group) is the "
        "Phase-15 whole-run definition (`mean_immediate_satisfaction / (mean_watch_seconds / "
        "60)` unless package A provides a figure directly); the session-health panel's "
        "`satisfaction per minute (session-scoped)` is a DIFFERENT aggregation -- per CLOSED "
        "SESSION rather than pooled over the whole run -- and is expected to diverge from it "
        "once real, especially for arms with many short regret-truncated sessions.\n"
        "- **Oracle arm**: unlike Phase 15 (where it was pending package B2), `oracle_satisfaction` "
        "is a Phase 15 deliverable and is expected to already work in a Phase 16 worktree; a "
        "missing oracle column here means the arm was not run / not passed on the command line, "
        "not a pending-integration stub.\n"
        "- **Pre-integration `n/a` cells outside session health**: none expected -- "
        "`comment_rate`/`save_rate`/`profile_visit_rate` and per-archetype exposure "
        "(`welfare_archetype_metrics.csv`) are Phase 15 deliverables and should already be "
        "populated. If any of those still read `n/a (pre-integration)` in a report generated from "
        "this script, that is a regression worth investigating, not an expected Phase 16 gap.\n"
    )
    return "\n".join(parts) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Phase 16 four-arm session-dynamics re-run (V2 TDD §4.6-4.9, §6) "
                    "as comparison.csv + comparison.md.",
    )
    parser.add_argument("--semantic", required=True, type=Path,
                        help="semantic (hnsw) arm result directory")
    parser.add_argument("--engagement", required=True, type=Path,
                        help="engagement-optimized (hnsw_ranker + engagement preset) arm result directory")
    parser.add_argument("--proxy", required=True, type=Path,
                        help="satisfaction-proxy (hnsw_ranker + proxy preset) arm result directory")
    parser.add_argument("--oracle", type=Path, default=None,
                        help="oracle_satisfaction arm result directory (optional)")
    parser.add_argument("--out", type=Path, default=Path("results/published/phase16"),
                        help="output directory for comparison.csv/comparison.md "
                             "(default: results/published/phase16)")
    parser.add_argument("--precision", type=int, default=6,
                        help="significant figures for floats (default: 6)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    arms = load_arms(args)

    required_available = [arms[n].available for n in ("semantic", "engagement", "proxy")]
    if not any(required_available):
        p15.warn("no readable summary.json in any of the required arms (semantic/engagement/proxy)")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    rows = p15.build_rows(arms, args.precision)

    csv_path = args.out / "comparison.csv"
    p15.write_csv(csv_path, rows)

    md_path = args.out / "comparison.md"
    md_path.write_text(render_markdown(arms, rows, args.precision))

    print(f"phase16_comparison: wrote {csv_path}")
    print(f"phase16_comparison: wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
