# Performance/benchmark executables (design decision D7: never unit tests, excluded from
# ctest). Phase 1 adds benchmark_retrieval.

# benchmark_retrieval — HNSW-vs-exact retrieval sweep (Phase 1, task 4).
# Links only rr_vindex (which PUBLIC-links rr_core + nlohmann_json). D2 containment: this target
# does NOT link vector_db::vdb_core and includes no vector-db header — everything is reached
# through the rr:: VectorIndex adapters.
add_executable(benchmark_retrieval benchmark_retrieval.cpp)
target_link_libraries(benchmark_retrieval PRIVATE rr_vindex rr_warnings)

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
