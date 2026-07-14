# Phase 11 — Concurrent-read safety of the frozen HNSW index (D13, plan task 1)

**Verdict: PASS.** Concurrent `const` search on a fully-built, frozen `HNSWVectorIndex` is
data-race-free on this toolchain. The Phase 11 load driver's design — **one shared read-only index,
per-thread everything-else** — is sound; **no per-thread index replicas and no mutex are needed**
(D13's fallback is not triggered).

Three independent lines of evidence agree: (1) static inspection of the vector-db search path, (2) a
clean ThreadSanitizer run, (3) a Release stress + determinism cross-check.

- Hardware/OS: Apple M5, 10 cores, 24 GB, macOS (Darwin 25.5.0), AppleClang.
- vector-db HEAD: `17e434a` (read-only reference). reel-rank: Phase 11 package A worktree.

---

## 1. Static inspection (re-verified against the current vector-db source)

The one object shared across threads is the `HNSWVectorIndex`. Its search path was read in full and
confirmed to touch **no shared mutable state**; the only writers are `insert()` and `setEfSearch()`,
neither of which runs during the concurrent window (the index is frozen first).

| Claim | Evidence (file:line) |
|---|---|
| `HNSWIndex::search(...) **const**` is the entry point | `vector-db/src/algorithms/hnsw_index.hpp:62`; impl `hnsw_index.cpp:226-250` |
| `search()` reads only `dimensions_`, `entry_point_`, `max_level_`, `ef_search_`, `nodes_`; all other state is function-local (`entry`, `ef`, `ranked`, `results`) | `hnsw_index.cpp:226-250` |
| `searchLayer()` const; its visited-set is a **per-call local** `std::vector<uint8_t> visited(nodes_.size(), 0)`; `candidates`/`results` are local heaps | `hnsw_index.cpp:84-119` (local declared at `:86`) |
| `greedyDescent()` const; only local `current`/`current_dist`, reads `nodes_` | `hnsw_index.cpp:63-82` |
| `distance()` const → `metric_->distance(a,b)`; `metric_` is `std::shared_ptr<**const** DistanceMetric>` (const-only call, no shared_ptr copy → no refcount write) | `hnsw_index.cpp:42-44`; member decl `hnsw_index.hpp:98` |
| `EuclideanDistance::distance()` const **and stateless** (only a local `squared`) | `vector-db/src/utils/distance_metrics.cpp:7-23` |
| The engine `rng_` is read **only** by `randomLevel()`; `search`/`greedyDescent`/`searchLayer` never reference it | `hnsw_index.cpp:54-61` (randomLevel); `hnsw_index.hpp:106-107` comment "Only insert() touches it" — confirmed by the const-impl reads above |
| No `mutable` members exist (entire private section inspected) | `hnsw_index.hpp:80-122` |
| The only writers of `nodes_`, `key_to_id_`, `entry_point_`, `max_level_`, `rng_` are `insert()` (not called concurrently) … | `hnsw_index.cpp:159-225` |
| … and `setEfSearch()` mutates `ef_search_` (**never called while threads search** — the driver sets ef via config before freezing) | `hnsw_index.cpp:252-254` |
| The rr adapter `HNSWVectorIndex::search()` is const and allocates only locals (`::Vector q`, `results`); `std::stoul` on the key uses thread-local `errno` only | `reel-rank/src/vindex/hnsw_vector_index.cpp:64-76` |

**Conclusion of inspection:** every write to index state is confined to `insert()`/`setEfSearch()`.
Once the graph is built and neither is called again, concurrent `search()` is pure reads over
immutable state plus function-local scratch → no data race is possible by construction.

## 2. ThreadSanitizer evidence

Built in a dedicated TSan tree and run there. TSan detects a data race as an unsynchronized pair of
accesses where at least one is a write; a clean report on a workload with **zero writers** to shared
state is direct proof there is no hidden shared write on the search path (mutable scratch, lazy
cache, atomic refcount churn, etc.).

Build (worktree):
```
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread \
  -DREELRANK_VDB_DIR=/abs/path/to/vector-db
cmake --build build-tsan --target concurrency_check benchmark_recommender
```

### 2a. `concurrency_check` — the shared index in isolation
`concurrency_check` (links `rr_vindex` only, D2) builds a frozen 8,000×64d index, then runs 8 threads
issuing 500 `const` searches each (4,000 total): a mix of **distinct per-thread query streams** and a
set of **shared identical queries** cross-checked against a single-threaded reference computed before
any thread starts.

```
TSAN_OPTIONS="halt_on_error=0 exitcode=66" \
  ./build-tsan/apps/concurrency_check --vectors 8000 --dim 64 --threads 8 --searches 500 --k 100
```
Result (full log: `tsan_concurrency_check.log`): **exit 0**, `WARNING: ThreadSanitizer` count = **0**,
`malformed results: 0`, `determinism mismatches: 0`. The determinism cross-check confirms concurrent
searches return byte-identical id lists to the single-threaded reference — no logical corruption.

### 2b. `benchmark_recommender --smoke` — the full pipeline under TSan
`concurrency_check` links only `rr_vindex`, so it cannot reach the orchestrator / candidate sources
(those live in `rr_recommend`). The **full-pipeline** concurrency — 6 per-thread candidate sources +
ranker + reranker + orchestrator all sharing the one frozen index and the shared `const` reels/users
vectors — is therefore verified by running the **actual load driver** under the same TSan tree, which
is strictly stronger evidence than re-implementing the arrangement in the probe:

```
TSAN_OPTIONS="halt_on_error=0 exitcode=66" ./build-tsan/apps/benchmark_recommender --smoke
```
Result (full log: `tsan_benchmark_recommender_smoke.log`): **exit 0**, `WARNING: ThreadSanitizer`
count = **0**, across T = 1, 2, 4 issuing full `recommend()` calls. This exercises the per-thread
mutable state (Popular/Trending/Fresh/Creator `scratch_` buffers, the Exploration `Rng*` +
`lastFiredSlots_`) and confirms it is genuinely per-thread — no cross-thread aliasing.

## 3. Release stress + determinism cross-check

The plain `-O3` Release build (`build-rel`) was run as a non-instrumented stress pass:
- `concurrency_check --smoke` (Release): 1,200 concurrent searches / 4 threads, 0 malformed, 0
  mismatches, exit 0.
- `benchmark_recommender --smoke` (Release): its built-in content-determinism check runs the whole
  8-source pipeline twice with the same seed across 4 threads and asserts **byte-identical feed
  contents** (timing differs); PASS. A race would surface here as divergent feeds.

Additionally, the T=8 profile (`../profiling/sample_100k_t8.txt`) shows the 8 worker threads spend
their time in the search/scoring/ranking code with `__ulock_wait` accounting for roughly one thread's
worth of samples (the main thread blocked on `join`) — i.e. **no lock contention among the workers**,
consistent with a lock-free shared-read design.

---

## Decision recorded (D13)

The frozen-index concurrent-read assumption holds. The load driver uses **one shared read-only
`HNSWVectorIndex`** with **per-thread** candidate sources, ranker, reranker, orchestrator, and
exploration `Rng`. No mutex is added to the index and no per-thread replica is built. The index must
be **fully built before any thread starts**, and **`setEfSearch()`/`insert()` must never be called
during the concurrent window** — the driver enforces both (index frozen after construction; ef set
from config before freezing).

### Deviation noted for the phase record
Task 1 as written asked `concurrency_check` to also run a "phase 2" full-pipeline arrangement. The
frozen CMake wiring links `concurrency_check` against `rr_vindex` **only** (by design — it is the
minimal probe for the one shared object, and D2 keeps it free of the recommender libraries), so the
orchestrator/candidate-source symbols are not linkable there. The full-pipeline concurrency check is
delivered instead by building and running `benchmark_recommender --smoke` in the same TSan tree
(§2b) — the real load-driver arrangement, which is stronger evidence than a duplicate. No CMake was
modified.
