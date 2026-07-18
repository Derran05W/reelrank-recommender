# V1 golden baseline (D17)

This directory is the byte-identity anchor for the entire Realism V2 upgrade (Phases 13–24,
`plan/00-DESIGN-DECISIONS-V2.md` D17). Every V2 mechanism ships behind a config gate that
defaults OFF; the standing exit criterion for every V2 phase is that a gates-off Release run
reproduces what's committed here **byte-identically**. `scripts/check_golden.py` is the tool that
proves it — see "How to run the check" below. This README is the tool's companion documentation:
what was captured, how it's compared, and why certain files/fields are excluded.

## What's here

Two arms, each a full experiment output directory (minus a few intentionally-excluded files —
see "Comparison rules"):

```
tests/golden/v1-baseline/
├── small/           # configs/small.json          under hnsw_ranker_diversity
│   ├── diversity_metrics.csv
│   ├── learning_curve.csv
│   ├── recommendation_metrics.csv
│   ├── regret_curve.csv
│   ├── retrieval_metrics.csv
│   ├── summary.json
│   ├── config.json
│   └── metadata.json
└── drift-medium/    # configs/phase10-drift.json  under hnsw_ranker (Phase 10 drift arm)
    └── (same 8 files)
```

## Capture provenance

Captured once, at the start of Phase 13, at the clean V1-complete commit (before any Phase 13
scaffolding existed):

| | |
|---|---|
| reel-rank SHA | `c8e032b0620e419930bf01bbc076778bf450c3d7` (short `c8e032b`), clean working tree |
| vector-db SHA | `17e434a3e741f702ffb8e0f00b0484676d988198` (short `17e434a`) |
| Build type | Release |
| Compiler | AppleClang 21.0.0.21000101 |
| OS / arch | Darwin 25.5.0, arm64 (Apple M5, 10 cores) |
| Date | 2026-07-18 |

Exact commands used to capture (from the reel-rank repo root, Release binaries already built at
`c8e032b`):

```
build-release/apps/simulate --config configs/small.json --algorithm hnsw_ranker_diversity --out results/golden-capture/small
build-release/apps/simulate --config configs/phase10-drift.json --algorithm hnsw_ranker --out results/golden-capture/drift
```

Each writes `<out>/<algorithm>-seed42-<timestamp>/`; the resulting directory's contents (minus
the excluded files below) were copied verbatim into `small/` and `drift-medium/` above.
`results/golden-capture/` itself is a scratch capture location (gitignored, D12 `results/`
convention) — it is not part of the committed golden and need not exist for the check to work.

### The two arm commands (canonical form, verbatim)

`scripts/check_golden.py --run` executes exactly these, with `<tmp>` a freshly created temp
directory:

- **small**: `build-release/apps/simulate --config configs/small.json --algorithm hnsw_ranker_diversity --out <tmp>`
- **drift-medium**: `build-release/apps/simulate --config configs/phase10-drift.json --algorithm hnsw_ranker --out <tmp>`

Each is a single-threaded Release run (D13); expect roughly 6 minutes per arm on this machine
(the drift-medium arm is the Phase 10 10k-user/100k-reel/20-round drift scenario, materially
larger than the small 1k-user/10k-reel/5-round arm). `scripts/check_golden.py` prints a note
before launching either one.

### The drift-medium `metadata.json` "dirty" quirk

`drift-medium/metadata.json` records `"reel_rank_dirty": true`, while `small/metadata.json`
records `false`. This is expected, not a provenance defect: the small arm was captured first
(00:12:47), before any Phase 13 file existed in the working tree; the drift-medium arm was
captured seven minutes later (00:19:07), by which point inert Phase 13 scaffolding edits
(new headers, stub `.cpp` files — nothing that touches any V1 code path) had landed in the
working tree. Critically, **both arms' binaries were built from `build-release/`, which was
compiled at clean `c8e032b` before either run started** — the scaffolding files that made the
tree "dirty" are not referenced by the Release build in any way, and the golden CSVs/summary
are what those clean binaries produced. The dirty flag is retained here unedited because
`metadata.json` is provenance (see "Comparison rules": it isn't compared by the tool anyway,
so the flag doesn't affect verification) and because honesty about working-tree state at
capture time is more useful than a doctored-clean record. The independent, ongoing proof of
byte-identity is exactly what `check_golden.py` re-verifies at every phase end: it does not
trust this flag, it re-derives the CSVs/summary from whatever the phase-HEAD gates-off build
produces and compares them to what's committed here.

## Comparison rules

These are implemented in `scripts/check_golden.py`; this section is the human-readable
explanation of *why* each rule is what it is.

### Compared: the five deterministic CSVs — BYTE-IDENTICAL required

`diversity_metrics.csv`, `learning_curve.csv`, `recommendation_metrics.csv`, `regret_curve.csv`,
`retrieval_metrics.csv`. These contain only simulation-derived, deterministic-given-(config,seed)
values (D8) — no wall-clock content of any kind — so a gates-off re-run must reproduce them as
literal identical bytes. Any difference, however small (a single flipped digit, a trailing
newline), is a FAIL.

### Compared: `summary.json` — parsed JSON, with timing-derived fields stripped

`summary.json` mixes deterministic metrics with wall-clock measurements (the D9 exception:
`rr::Stopwatch` is used *only* for latency measurement, never for simulation logic — but that
means its output legitimately lands in this file). The check parses both sides as JSON, removes
the fields below, and requires the remainder to be **exactly equal** (bit-equal numbers — Python
`==` on values parsed by the standard `json` module, which is exact double-precision comparison,
not a tolerance-based comparison). The strip list, kept as the `SUMMARY_JSON_STRIP_PATHS` constant
in `scripts/check_golden.py` (edit both places if it ever changes):

| Path | Why it's stripped |
|---|---|
| `experiment_id` | Embeds the wall-clock capture timestamp in `<algorithm>-seed<seed>-<yyyyMMDDThhmmss>` (the run's own output-directory name) — a byte-identical re-run still mints a fresh timestamp, so this string can never match across two separate runs. |
| `timing.total_wall_seconds` | Wall-clock total experiment runtime (`rr::Stopwatch`, D9) — varies run to run even given byte-identical simulation output. |
| `timing.retrieval_latency_ms` | Wall-clock retrieval-stage latency percentiles (p50/p95/p99/max/mean over N samples) — hardware/scheduler dependent, not a function of simulated content. |
| `timing.ranking_latency_ms` | Wall-clock ranking-stage latency percentiles — hardware/scheduler dependent. |
| `timing.reranking_latency_ms` | Wall-clock reranking-stage latency percentiles — hardware/scheduler dependent. |
| `timing.recommend_latency_ms` | Wall-clock end-to-end `recommend()` latency percentiles — hardware/scheduler dependent. |

Everything else in `summary.json` — `algorithm`, `seed`, `counts`, `diversity`, `learning`,
`metrics`, `notes`, `oracle`, `retrieval`, and (drift-medium only) `adaptation` — is a
deterministic function of `(config, seed)` and must compare bit-equal.

### NOT compared

- **`latency_metrics.csv`** — pure wall-clock timing (same content as `summary.json`'s `timing`
  block, in per-stage-percentile CSV form). Not committed to the golden directories at all (only
  the five deterministic CSVs + `summary.json` + `config.json` + `metadata.json` are committed);
  if a run directory has it, the tool ignores it.
- **`config.json`** — the fully-resolved config the run actually used. Excluded because config is
  *additive by design* (D6: unknown keys are a load error, but new keys ship with defaults, so a
  later-phase run's fully-resolved config can legitimately gain keys — e.g. the whole `realism`
  block — that the V1 golden capture predates). Comparing it would force the golden to be
  re-captured on every additive config change, defeating the point of a stable anchor.
- **`metadata.json`** — run provenance (git SHA, host/CPU/OS, build type, timestamps, dataset
  scale). Expected to differ on every invocation by construction; it documents *how a run was
  produced*, not *what it produced*. See the "dirty quirk" note above for why this file's content
  is trusted for provenance but never diffed for verification.

## The Release-vs-Release, same-machine rule (D17)

This golden check is a **this-machine, phase-end gate**, not a CI gate, and it always compares
Release-built binaries against Release-built binaries:

- The main `build/` directory on this machine is configured **Debug**; Debug and Release float
  codegen can legally differ (fast-math-adjacent optimizations, contraction, etc.), so a Debug
  run is not expected to match this golden and should never be compared against it. Always use
  `build-release/` (or an equivalent Release configuration) when checking against this baseline.
- Cross-platform byte-identity is **not** promised, even in Release. `rr::Rng`'s integer and
  uniform samplers are bit-identical across platforms by construction (D8), but `gaussian()`
  (Box–Muller over libm `log`/`cos`) can differ by a few ULPs across C libraries — this was
  already documented at V1 Phase 0 (`docs/LIMITATIONS.md`: "`gaussian()` golden tests use a
  1e-12 tolerance, not bit-exactness... strict cross-libm bit-identity of gaussian draws is not
  guaranteed"). The CSV/summary byte-identity promise here is therefore scoped to **this machine,
  this toolchain** (AppleClang on this Mac) reproducing itself — it is deliberately not wired into
  the cross-platform CI matrix (macOS + Ubuntu) for the same reason V1's `gaussian()` regression
  test uses a tolerance instead of exact equality.

In short: run the check on the same machine/toolchain that owns `build-release/`, always Release,
and treat a pass as "this phase's gates-off code path did not change V1 behavior on this machine,"
not as a claim of universal bit-for-bit portability.

## How to run the check

```
# Compare an existing run directory (either the leaf run dir or its --out root) against golden:
scripts/check_golden.py --arm small --run-dir <path> [--repo <reel-rank root>]
scripts/check_golden.py --arm drift-medium --run-dir <path>

# Execute the arm fresh (~6 minutes) into a temp dir, then compare:
scripts/check_golden.py --arm small --run
scripts/check_golden.py --arm drift-medium --run
```

Prints one `PASS`/`FAIL` line per compared file (with a short unified-diff excerpt, first ~20
lines, on any `FAIL`) plus a `SKIP` line per excluded file, and exits non-zero if anything failed.
D17 mandates the small-config check at every V2 phase's exit criteria; the drift-medium check is
mandatory at Phases 14, 16, 18, and 24 (the phases that touch the behaviour/runner core) and
discretionary elsewhere.

## Regeneration instructions

This baseline should not normally need recapturing — it anchors V1 semantics permanently. The
only legitimate reason to redo this is an intentionally-approved, reviewed change to V1's
gates-off behavior itself (which should be rare and heavily scrutinized, since D17's entire point
is that gates-off behavior does NOT change across the V2 phases). If that ever happens:

1. Build a clean Release tree at the SHA the new golden should represent:
   `git status` must be clean; `cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
   -DREELRANK_VDB_DIR=/Users/derranw/vector-db && cmake --build build-release -j8`.
2. Run both arm commands exactly as documented above (or `scripts/check_golden.py --arm <arm>
   --run` twice, once per arm, and locate the temp output roots it prints).
3. Replace this directory's contents: for each arm, copy the five deterministic CSVs,
   `summary.json`, `config.json`, and `metadata.json` from the fresh run into
   `tests/golden/v1-baseline/<arm>/`, overwriting what's there.
4. Update this README's "Capture provenance" table (new SHAs, date, hardware) and re-check the
   "dirty quirk" note still applies (or remove it if the tree was clean for both captures this
   time).
5. Sanity-check the new golden against itself: `scripts/check_golden.py --arm <arm> --run-dir
   tests/golden/v1-baseline/<arm>` should trivially PASS (comparing a directory to itself).
6. Commit the updated golden + README together, with a message explaining the approved V1
   behavior change that motivated the recapture.
