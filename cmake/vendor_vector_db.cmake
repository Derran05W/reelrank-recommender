# vector-db's CMakeLists.txt derives all paths from ${CMAKE_SOURCE_DIR}, so it cannot be
# consumed via add_subdirectory without modifying it (forbidden by design decision D2).
# We build vdb_core ourselves from the checkout, mirroring upstream's target
# (sources, PUBLIC include dirs, nlohmann_json PUBLIC link, x86-64 SIMD flags).
set(VDB_SRC ${REELRANK_VDB_DIR}/src)

add_library(vdb_core STATIC
    ${VDB_SRC}/core/vector.cpp
    ${VDB_SRC}/core/vector_database.cpp
    ${VDB_SRC}/core/kd_tree.cpp
    ${VDB_SRC}/algorithms/hnsw_index.cpp
    ${VDB_SRC}/algorithms/lsh_index.cpp
    ${VDB_SRC}/algorithms/approximate_nn.cpp
    ${VDB_SRC}/optimizations/parallel_processing.cpp
    ${VDB_SRC}/optimizations/simd_operations.cpp
    ${VDB_SRC}/utils/distance_metrics.cpp
    ${VDB_SRC}/utils/random_generator.cpp
    ${VDB_SRC}/features/atomic_batch_insert.cpp
    ${VDB_SRC}/features/atomic_file_writer.cpp
    ${VDB_SRC}/features/atomic_persistence.cpp
    ${VDB_SRC}/features/commit_log.cpp
    ${VDB_SRC}/features/query_cache.cpp
    ${VDB_SRC}/features/recovery_state_machine.cpp
)
target_include_directories(vdb_core SYSTEM PUBLIC
    ${VDB_SRC}/core
    ${VDB_SRC}/utils
    ${VDB_SRC}/algorithms
    ${VDB_SRC}/optimizations
    ${VDB_SRC}/features
    ${REELRANK_VDB_DIR}
)
target_link_libraries(vdb_core PUBLIC nlohmann_json::nlohmann_json)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
    target_compile_options(vdb_core PRIVATE -mavx2 -mfma)
endif()
add_library(vector_db::vdb_core ALIAS vdb_core)
