# ReelRank

ReelRank is a single-process C++20 simulation and evaluation harness for short-video ("reels")
recommendation. It generates a synthetic world of topics, creators, reels, and users with hidden
preferences, simulates their behaviour, runs pluggable recommendation pipelines (random,
popularity, exact-vector, and HNSW-based retrieval with ranking, diversity, and exploration), and
measures retrieval, ranking, behaviour, and diversity metrics — all deterministically from a
single master seed. Approximate nearest-neighbour search is provided by the sibling
[`vector-db`](../vector-db) library.

## Prerequisites

- CMake >= 3.20
- A C++20 compiler (AppleClang / Clang / GCC)
- A sibling `vector-db` checkout (default location `../vector-db`; override with
  `-DREELRANK_VDB_DIR=/path/to/vector-db`)
- Network access on first configure (FetchContent pulls nlohmann/json v3.11.3 and GoogleTest
  v1.15.2)

## Build and test

Debug:

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
ctest --test-dir build-debug --output-on-failure
```

Release:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
ctest --test-dir build-release --output-on-failure
```

Point at a non-default vector-db checkout:

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DREELRANK_VDB_DIR=/abs/path/to/vector-db
```

## vector-db integration (shadow build)

vector-db's `CMakeLists.txt` derives all of its source paths from `${CMAKE_SOURCE_DIR}`, so it
cannot be consumed via `add_subdirectory` without editing it — and vector-db is out of scope for
ReelRank changes. Instead, `cmake/vendor_vector_db.cmake` builds the `vdb_core` static library
ourselves directly from the checkout, mirroring upstream's target exactly (source list, PUBLIC
include dirs, `nlohmann_json` PUBLIC link, and x86-64 SIMD flags). It is exposed as
`vector_db::vdb_core`.

Because we never include vector-db's own `CMakeLists.txt`, the upstream `VDB_BUILD_TESTS`,
`VDB_BUILD_BENCHMARKS`, and `VDB_BUILD_SERVER` options are moot — no httplib download, no stray
`basic_usage` target, and no second copy of GoogleTest in the build tree. Vendored headers are
included as `SYSTEM`, keeping them out of ReelRank's `-Werror` warning set.

Per design decision D2, vector-db symbols (global namespace: `Vector`, `HNSWIndex`, ...) are only
ever touched by the adapters under `src/vindex/` (Phase 1) and the link-sanity integration test.
The rest of ReelRank sees only the `rr::VectorIndex` abstraction.

## Layout

- `include/rr/`, `src/` — headers and implementations (namespace `rr`)
- `configs/` — `small`, `medium`, `large`, `benchmark` experiment configs (JSON)
- `tests/` — `unit`, `integration`, `property`, `differential` (one `reel_rank_tests` binary)
- `apps/` — simulation and benchmark executables (later phases)
- `scripts/` — Python (uv) result analysis tooling
- `results/published/` — curated, committed benchmark reports (rest of `results/` is gitignored)
