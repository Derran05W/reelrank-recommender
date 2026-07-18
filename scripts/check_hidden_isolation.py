#!/usr/bin/env python3
"""D18 hidden-state include-graph guard (Realism V2, Phase 13 — runs under ctest and CI).

Hidden simulator-owned truth lives in headers under include/rr/simulation/hidden/ (D18). The
recommender side must never reach one of those headers, even transitively: this script walks the
quoted-include graph of every file under the recommender-side roots

    include/rr/recommendation  include/rr/candidate_sources  include/rr/learning
    src/recommendation         src/candidate_sources         src/learning

and exits non-zero, printing the offending include chain, if any of them reaches a header whose
repo-relative path contains "simulation/hidden/". Evaluation modules are deliberately NOT
scanned: they are the documented carve-out (metrics and explicitly-labeled oracle arms may read
hidden state, D18).

Usage: check_hidden_isolation.py <repo-root>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_ROOTS = [
    "include/rr/recommendation",
    "include/rr/candidate_sources",
    "include/rr/learning",
    "src/recommendation",
    "src/candidate_sources",
    "src/learning",
]

HIDDEN_MARKER = "simulation/hidden/"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def resolve_include(repo: Path, inc: str) -> Path | None:
    """Map a quoted include to a repo file (project convention: quoted includes are rr/...
    paths rooted at include/)."""
    candidate = repo / "include" / inc
    if candidate.is_file():
        return candidate
    return None


def quoted_includes(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # unreadable source is a guard failure, not a silent pass
        sys.exit(f"check_hidden_isolation: cannot read {path}: {exc}")
    return INCLUDE_RE.findall(text)


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: check_hidden_isolation.py <repo-root>")
    repo = Path(sys.argv[1]).resolve()
    if not (repo / "include" / "rr").is_dir():
        sys.exit(f"check_hidden_isolation: {repo} does not look like the reel-rank root")

    # parent[header] = the file that first included it, for chain reconstruction.
    violations = []
    scanned = 0
    for root in FORBIDDEN_ROOTS:
        root_path = repo / root
        if not root_path.is_dir():
            continue
        for start in sorted(root_path.rglob("*")):
            if start.suffix not in {".hpp", ".cpp"}:
                continue
            scanned += 1
            stack = [start]
            parent: dict[Path, Path] = {}
            seen = {start}
            while stack:
                current = stack.pop()
                for inc in quoted_includes(current):
                    if HIDDEN_MARKER in inc:
                        chain = [inc]
                        node = current
                        while node is not None:
                            chain.append(str(node.relative_to(repo)))
                            node = parent.get(node)
                        violations.append(" <- ".join(chain))
                        continue
                    target = resolve_include(repo, inc)
                    if target is None or target in seen:
                        continue
                    seen.add(target)
                    parent[target] = current
                    stack.append(target)

    if violations:
        print(f"D18 VIOLATION: recommender-side code reaches simulation/hidden/ "
              f"({len(violations)} chain(s)):")
        for v in sorted(set(violations)):
            print(f"  {v}")
        return 1
    print(f"hidden-state isolation OK: {scanned} recommender-side files scanned, "
          f"no path reaches {HIDDEN_MARKER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
