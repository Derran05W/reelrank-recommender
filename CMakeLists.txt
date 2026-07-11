cmake_minimum_required(VERSION 3.20)
project(reel_rank VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- vector-db checkout location ------------------------------------------------------------
set(REELRANK_VDB_DIR ${CMAKE_SOURCE_DIR}/../vector-db CACHE PATH "Path to a vector-db checkout")
if(NOT EXISTS ${REELRANK_VDB_DIR}/src/algorithms/hnsw_index.hpp)
    message(FATAL_ERROR
        "vector-db checkout not found. Expected "
        "${REELRANK_VDB_DIR}/src/algorithms/hnsw_index.hpp to exist. "
        "Set -DREELRANK_VDB_DIR=/path/to/vector-db to a sibling checkout.")
endif()

# --- third-party dependencies ---------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

# --- vendored vector-db (shadow build; see cmake/vendor_vector_db.cmake) ---------------------
include(cmake/vendor_vector_db.cmake)

# --- warning flags carried into every ReelRank target ---------------------------------------
add_library(rr_warnings INTERFACE)
target_compile_options(rr_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)

# --- rr_core: infrastructure + core (never links or includes vector-db, per D2) -------------
add_library(rr_core STATIC
    src/core/embedding.cpp
    src/infrastructure/random.cpp
    src/infrastructure/logger.cpp
    src/infrastructure/config.cpp
)
target_include_directories(rr_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rr_core PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(rr_core PRIVATE rr_warnings)

# --- rr_vindex: VectorIndex implementations (Phase 1). D2 containment: vector-db headers are
# confined to src/vindex/*.cpp; the public headers under include/rr/vindex/ expose only rr::
# types, so vdb_core can stay a PRIVATE link here. -------------------------------------------
add_library(rr_vindex STATIC
    src/vindex/exact_vector_index.cpp
    src/vindex/hnsw_vector_index.cpp
)
target_include_directories(rr_vindex PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rr_vindex PUBLIC rr_core)
target_link_libraries(rr_vindex PRIVATE vector_db::vdb_core rr_warnings)

# --- rr_simulation: synthetic domain generation (Phase 2) + behaviour simulation and reward
# (Phase 3, incl. src/learning/). Never includes vector-db. -----------------------------------
file(GLOB RR_SIMULATION_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/simulation/*.cpp
    ${CMAKE_SOURCE_DIR}/src/learning/*.cpp)
add_library(rr_simulation STATIC ${RR_SIMULATION_SOURCES})
target_include_directories(rr_simulation PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rr_simulation PUBLIC rr_core)
target_link_libraries(rr_simulation PRIVATE rr_warnings)

add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
