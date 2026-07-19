# Event-log digest golden (Phase 18, D20)

The compact "same seed produces identical event sequence" tripwire for the event-driven runner
(V2 TDD §7 / D20). A single event-mode run of the pinned config here must reproduce a committed
digest of its event log — the cross-commit / cross-platform companion to the in-process
`tests/property/event_determinism_test.cpp` suite (which proves digest *equality across two runs*;
this golden proves equality against a *committed value*).

## Files

| File | Role |
|------|------|
| `config.json` | PINNED tiny full-gate event-mode config (seed 42, 200 users / 2000 reels / dim 32, `scheduler="event_queue"`, `horizon_seconds=21600` = 6h, `content_v2`+`latent_reactions`+`session_dynamics`, `hnsw_ranker`). Do not edit without regenerating the digest. |
| `digest.txt` | The golden `event_log_digest` + `event_count`, as `key=value` lines. **Currently a PLACEHOLDER** (`event_log_digest=PENDING_PACKAGE_A_REGENERATE`) until package A lands the runner. |
| `../../../scripts/check_event_digest.py` | Runs the fixture, extracts the digest from `summary.json`, compares to `digest.txt`. Degrades to PENDING pre-merge. |

## Status: PENDING package A

Package A owns `EventDrivenRunner`; in this scaffold it is a throwing stub, so any event-mode run
throws and `digest.txt` ships as a placeholder. `check_event_digest.py` and the C++ suite both detect
this and report PENDING (never a spurious failure). **Nothing here is a real check until A merges and
the digest is regenerated.**

## Regeneration (integrator, once, post-merge)

The digest is a deterministic function of `(config.json, seed)`. Package A exposes it on
`ExperimentResult` (`eventLogDigest`/`eventCount`) and in `summary.json` (D22). After A merges:

1. Build the Release tree (matches the D17 golden-baseline convention; the digest hashes the event
   *identity* sequence, so it is expected to be build-type-independent, but pin it with Release for
   consistency and confirm Debug agrees once):

   ```
   cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DREELRANK_VDB_DIR=/Users/derranw/vector-db
   cmake --build build-release -j
   ```

2. **Confirm the summary.json key.** This tooling reads the digest by *candidate key path*
   (`DIGEST_CANDIDATES` in `check_event_digest.py`, mirrored by `eventDigestFromSummary()` in
   `tests/property/event_determinism_test.cpp`). If A emitted the digest under a key not in that
   list, add A's actual key to BOTH lists (they are kept in sync by hand).

3. Pin the real digest:

   ```
   python3 scripts/check_event_digest.py --regenerate --repo .
   ```

   This runs the fixture and overwrites `digest.txt` with the harvested `event_log_digest` /
   `event_count`. Commit the result.

4. Verify the golden now passes:

   ```
   python3 scripts/check_event_digest.py --run --repo .
   ```

   Expect `RESULT: PASS`. Re-run on a second machine / CI to confirm cross-platform stability
   (the whole point of a committed digest).

5. **Wire it into CI/ctest** alongside `check_golden.py` and `check_hidden_isolation.py` (add an
   `add_test`/CI step invoking `check_event_digest.py --run`). Left unwired here because it is inert
   pre-merge; the integrator enables it in the same commit that regenerates the digest.

## Comparing an existing run without re-running

```
python3 scripts/check_event_digest.py --run-dir <path-to-run-or-output-root> --repo .
```

Accepts either the leaf run directory (the one with `summary.json`) or an output root containing a
single `hnsw_ranker-seed42-*` run.
