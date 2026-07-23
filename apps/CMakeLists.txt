# Performance/benchmark executables (design decision D7: never unit tests, excluded from
# ctest). Phase 1 adds benchmark_retrieval.

# benchmark_retrieval — HNSW-vs-exact retrieval sweep (Phase 1, task 4; Phase 11 extends the
# grid and adds a generator-backed clustered-data mode, hence rr_simulation). D2 containment:
# this target does NOT link vector_db::vdb_core and includes no vector-db header — everything is
# reached through the rr:: VectorIndex adapters.
add_executable(benchmark_retrieval benchmark_retrieval.cpp)
target_link_libraries(benchmark_retrieval PRIVATE rr_vindex rr_simulation rr_warnings)

# Provenance/build facts baked in for metadata.json (see benchmark_retrieval.cpp).
target_compile_definitions(benchmark_retrieval PRIVATE
    RR_REPO_DIR="${CMAKE_SOURCE_DIR}"
    RR_VDB_DIR="${REELRANK_VDB_DIR}"
    RR_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    RR_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

# inspect_user — dumps topic/creator/reel/user distribution summaries as JSON (Phase 2, task 7)
# and per-item ranking explanations via --explain-user (Phase 6, task 3). Links rr_evaluation
# (PUBLIC-links rr_recommend + rr_simulation) for cold-start init and the recommender factory.
# D2 containment holds: no vector-db header here.
add_executable(inspect_user inspect_user.cpp)
target_link_libraries(inspect_user PRIVATE rr_evaluation rr_warnings)

# benchmark_recommender — Phase 11 multi-threaded closed-loop load driver (plan task 2, D13).
# Links rr_evaluation (PUBLIC-links rr_recommend + rr_simulation) for generators + the full
# pipeline components, and Threads for std::thread on Linux. D2 containment holds: no vector-db
# header here.
find_package(Threads REQUIRED)
add_executable(benchmark_recommender benchmark_recommender.cpp)
target_link_libraries(benchmark_recommender PRIVATE rr_evaluation Threads::Threads rr_warnings)
target_compile_definitions(benchmark_recommender PRIVATE
    RR_REPO_DIR="${CMAKE_SOURCE_DIR}"
    RR_VDB_DIR="${REELRANK_VDB_DIR}"
    RR_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    RR_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

# concurrency_check — Phase 11 task 1 (D13) TSan probe for concurrent const HNSW search on a
# frozen index. Build it in a dedicated -fsanitize=thread tree; excluded from ctest (D7).
add_executable(concurrency_check concurrency_check.cpp)
target_link_libraries(concurrency_check PRIVATE rr_vindex Threads::Threads rr_warnings)

# train_models — Phase 22 offline learner CLI (contracts §5, D21). SCAFFOLD STUB: the target is
# reserved so package B's worktree needs no CMake edit; B replaces train_models.cpp's body with the
# real trainer. Links rr_learning_v2 (the SGD learners + CSV training-data reader B lands there,
# PUBLIC-linking rr_core for config/random/nlohmann). D2 containment holds: no vector-db header.
add_executable(train_models train_models.cpp)
target_link_libraries(train_models PRIVATE rr_learning_v2 rr_warnings)

# simulate — end-to-end evaluation harness CLI (Phase 4, task 6). Links rr_evaluation (which
# PUBLIC-links rr_recommend + rr_simulation). D2 containment holds: no vector-db header here —
# vectors are reached through the rr:: adapters. Provenance/build facts baked in for metadata.json.
add_executable(simulate simulate.cpp)
target_link_libraries(simulate PRIVATE rr_evaluation rr_warnings)
target_compile_definitions(simulate PRIVATE
    RR_REPO_DIR="${CMAKE_SOURCE_DIR}"
    RR_VDB_DIR="${REELRANK_VDB_DIR}"
    RR_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    RR_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)
