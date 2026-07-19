#!/usr/bin/env python3
"""check_event_digest.py — Phase 18 event-log digest golden (D20).

Proves that an event-mode run of ReelRank on the PINNED tiny config
(``tests/golden/event-digest/config.json``) reproduces the committed event-log digest
(``tests/golden/event-digest/digest.txt``). This is the compact "same seed produces identical event
sequence" tripwire from V2 TDD §7 / D20, complementing the in-process
tests/property/event_determinism_test.cpp suite (which asserts digest EQUALITY across two runs; this
script asserts equality against a COMMITTED value, catching cross-commit / cross-platform drift).

PACKAGE-A DEPENDENCY. The event-log digest is exposed by package A's EventDrivenRunner (an additive
``eventLogDigest``/``eventCount`` pair on ExperimentResult and in summary.json, D22). Until A lands,
``apps/simulate`` throws on any event-mode run, and the committed digest.txt is a PLACEHOLDER
(sentinel ``PENDING_PACKAGE_A_REGENERATE``). This script degrades gracefully in BOTH cases: it
reports PENDING (exit 0) rather than a spurious failure. Post-merge the integrator runs
``--regenerate`` once to pin the real digest, after which ``--run`` is a hard check.

Usage:
    check_event_digest.py --run [--repo <root>] [--build-dir <dir>]
    check_event_digest.py --run-dir <path> [--repo <root>]
    check_event_digest.py --regenerate [--repo <root>] [--build-dir <dir>]

  --run         Run the fixture config through the event-mode simulate binary into a temp dir, then
                compare the harvested (digest, event_count) against digest.txt.
  --run-dir     Compare an EXISTING run directory's summary.json instead of launching a run.
  --regenerate  Run the fixture and OVERWRITE digest.txt with the harvested values (integrator step,
                post-merge). Fails if the runner is still a stub.
  --repo        reel-rank repo root (default: ".").
  --build-dir   Build tree holding apps/simulate (default: try build-release then build).

Exit status: 0 on PASS or PENDING; non-zero only on a genuine MISMATCH or unexpected error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

GOLDEN_DIR_REL = Path("tests") / "golden" / "event-digest"
FIXTURE_CONFIG_REL = GOLDEN_DIR_REL / "config.json"
DIGEST_FILE_REL = GOLDEN_DIR_REL / "digest.txt"

# The event-mode config sets simulation.scheduler="event_queue"; the run leaf dir is named
# "<algorithm>-seed<seed>-<timestamp>". The fixture pins algorithm=hnsw_ranker, seed=42.
RUN_GLOB = "hnsw_ranker-seed42-*"

PLACEHOLDER_SENTINEL = "PENDING_PACKAGE_A_REGENERATE"

# Signatures that identify package A's throwing stub (vs a genuine run failure), matched against the
# simulate binary's stderr ("error: <what()>").
STUB_SIGNATURES = ("not implemented", "package a stub", "eventdrivenrunner")

# Candidate summary.json key paths for the digest + event count, most-specific first. A's exact key
# is unknown pre-merge; keep this in sync with eventDigestFromSummary() in
# tests/property/event_determinism_test.cpp. (block, digest_key, count_key); "" block == top level.
DIGEST_CANDIDATES: list[tuple[str, str, str]] = [
    ("", "event_log_digest", "event_count"),
    ("event_log", "digest", "event_count"),
    ("event_log", "digest", "count"),
    ("event_log", "digest", "events"),
    ("determinism", "event_log_digest", "event_count"),
    ("event_mode", "event_log_digest", "event_count"),
    ("event_mode", "digest", "event_count"),
]


def find_simulate(repo: Path, build_dir: Optional[str]) -> Optional[Path]:
    """Locate the simulate binary. The digest is deterministic regardless of build type, but we
    prefer the Release tree to match the D17 golden-baseline convention (check_golden.py)."""
    if build_dir:
        cand = repo / build_dir / "apps" / "simulate"
        return cand if cand.is_file() else None
    for bd in ("build-release", "build"):
        cand = repo / bd / "apps" / "simulate"
        if cand.is_file():
            return cand
    return None


def run_fixture(repo: Path, simulate_bin: Path) -> tuple[int, str, Optional[Path]]:
    """Run the fixture config into a fresh temp root. Returns (returncode, combined_output, run_dir).
    run_dir is None when the run did not produce output."""
    tmp_root = Path(tempfile.mkdtemp(prefix="check_event_digest_"))
    cmd = [
        str(simulate_bin),
        "--config",
        str(repo / FIXTURE_CONFIG_REL),
        "--out",
        str(tmp_root),
    ]
    print(f"[check_event_digest] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    run_dir: Optional[Path] = None
    if proc.returncode == 0:
        run_dir = find_run_dir(tmp_root)
    return proc.returncode, combined, run_dir


def find_run_dir(root: Path) -> Optional[Path]:
    if (root / "summary.json").is_file():
        return root
    candidates = sorted(p for p in root.glob(RUN_GLOB) if p.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    return None


def is_stub_failure(combined_output: str) -> bool:
    low = combined_output.lower()
    return any(sig in low for sig in STUB_SIGNATURES)


def extract_digest(summary_path: Path) -> tuple[Optional[str], Optional[int]]:
    """Extract (digest_text, event_count) from summary.json by candidate key path. Reads defensively:
    a string or integer digest is normalized to text; missing => (None, None)."""
    if not summary_path.is_file():
        return None, None
    doc = json.loads(summary_path.read_text(encoding="utf-8"))

    def as_text(v: Any) -> Optional[str]:
        if isinstance(v, str):
            return v
        if isinstance(v, bool):  # guard: bool is an int subclass
            return None
        if isinstance(v, int):
            return str(v)
        return None

    for block, dkey, ckey in DIGEST_CANDIDATES:
        node = doc if block == "" else doc.get(block)
        if not isinstance(node, dict) or dkey not in node:
            continue
        digest = as_text(node.get(dkey))
        if digest is None:
            continue
        count_val = node.get(ckey)
        count = count_val if isinstance(count_val, int) and not isinstance(count_val, bool) else None
        return digest, count
    return None, None


def read_golden(digest_file: Path) -> tuple[Optional[str], Optional[int]]:
    """Parse the committed digest.txt ('key=value' lines; '#' comments and blanks ignored)."""
    digest: Optional[str] = None
    count: Optional[int] = None
    for raw in digest_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "event_log_digest":
            digest = value
        elif key == "event_count":
            try:
                count = int(value)
            except ValueError:
                count = None
    return digest, count


def write_golden(digest_file: Path, digest: str, count: Optional[int]) -> None:
    header = (
        "# ReelRank Phase 18 event-log digest golden (D20).\n"
        "#\n"
        "# Harvested by scripts/check_event_digest.py --regenerate from an event-mode run of\n"
        "# tests/golden/event-digest/config.json. Deterministic function of (config, seed); a change\n"
        "# here means the event sequence changed (a cross-commit/cross-platform determinism break)\n"
        "# unless the config or the runner was intentionally, reviewably changed. See README.md.\n"
        "#\n"
        '# Format: "key=value" lines; blank lines and \'#\' comments ignored.\n'
    )
    body = f"event_log_digest={digest}\nevent_count={count if count is not None else 0}\n"
    digest_file.write_text(header + body, encoding="utf-8")


def report_pending(reason: str) -> int:
    print(f"[check_event_digest] PENDING — {reason}")
    print("[check_event_digest] RESULT: PENDING (not a failure)")
    return 0


def compare(observed_digest: str, observed_count: Optional[int], repo: Path) -> int:
    golden_digest, golden_count = read_golden(repo / DIGEST_FILE_REL)
    if golden_digest == PLACEHOLDER_SENTINEL or golden_digest is None:
        return report_pending(
            f"digest.txt is still the placeholder. Observed digest={observed_digest!r} "
            f"event_count={observed_count}. Run `--regenerate` to pin it (integrator step)."
        )
    print(f"[check_event_digest] golden   digest={golden_digest!r} event_count={golden_count}")
    print(f"[check_event_digest] observed digest={observed_digest!r} event_count={observed_count}")
    ok = observed_digest == golden_digest
    if golden_count is not None and observed_count is not None:
        ok = ok and observed_count == golden_count
    print(f"[check_event_digest] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="run the fixture and compare to golden")
    group.add_argument("--run-dir", metavar="PATH", help="compare an existing run directory")
    group.add_argument(
        "--regenerate", action="store_true", help="run and overwrite digest.txt (integrator step)"
    )
    parser.add_argument("--repo", default=".", help="reel-rank repo root (default: '.')")
    parser.add_argument("--build-dir", default=None, help="build tree with apps/simulate")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / FIXTURE_CONFIG_REL).is_file():
        sys.exit(f"check_event_digest: fixture config not found: {repo / FIXTURE_CONFIG_REL}")

    # --run-dir: compare an existing run's summary.json, no binary needed.
    if args.run_dir:
        run_dir = find_run_dir(Path(args.run_dir).resolve())
        if run_dir is None:
            sys.exit(f"check_event_digest: no run dir with summary.json under {args.run_dir}")
        digest, count = extract_digest(run_dir / "summary.json")
        if digest is None:
            return report_pending(
                f"summary.json in {run_dir} exposes no event-log digest under any candidate key — "
                "package A must expose it, or sync DIGEST_CANDIDATES to A's key (see README.md)."
            )
        return compare(digest, count, repo)

    # --run / --regenerate need the binary.
    simulate_bin = find_simulate(repo, args.build_dir)
    if simulate_bin is None:
        where = args.build_dir or "build-release or build"
        sys.exit(
            f"check_event_digest: apps/simulate not found under {where} — build first "
            "(cmake --build build-release), see README.md"
        )

    returncode, output, run_dir = run_fixture(repo, simulate_bin)

    if returncode != 0:
        if is_stub_failure(output):
            reason = "EventDrivenRunner is still a stub (package A not merged)"
            if args.regenerate:
                sys.exit(f"check_event_digest: cannot regenerate — {reason}.")
            return report_pending(reason + ".")
        sys.exit(
            "check_event_digest: simulate failed unexpectedly (not the package-A stub):\n" + output
        )

    if run_dir is None:
        sys.exit("check_event_digest: run produced no summary.json (unexpected)")

    digest, count = extract_digest(run_dir / "summary.json")
    if digest is None:
        reason = (
            f"run succeeded but summary.json ({run_dir}) exposes no event-log digest under any "
            "candidate key — package A must expose it, or sync DIGEST_CANDIDATES (see README.md)."
        )
        if args.regenerate:
            sys.exit("check_event_digest: cannot regenerate — " + reason)
        return report_pending(reason)

    if args.regenerate:
        write_golden(repo / DIGEST_FILE_REL, digest, count)
        print(f"[check_event_digest] pinned digest={digest!r} event_count={count} into {DIGEST_FILE_REL}")
        print("[check_event_digest] RESULT: REGENERATED")
        return 0

    return compare(digest, count, repo)


if __name__ == "__main__":
    sys.exit(main())
