file(GLOB RR_TEST_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/differential/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/property/*.cpp
)

add_executable(reel_rank_tests ${RR_TEST_SOURCES})
target_link_libraries(reel_rank_tests
    PRIVATE
        rr_core
        rr_vindex
        rr_simulation
        rr_recommend
        rr_learning_v2
        rr_evaluation
        vector_db::vdb_core   # needed only by the vdb link-sanity TU
        GTest::gtest_main
        rr_warnings
)
target_compile_definitions(reel_rank_tests PRIVATE RR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(reel_rank_tests)

# D18 hidden-state include-graph guard (Phase 13): recommender-side modules must never reach a
# simulation/hidden/ header, even transitively. Registered as a plain ctest so it runs locally
# and in CI's existing ctest step.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME hidden_isolation_guard
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/check_hidden_isolation.py
                 ${CMAKE_SOURCE_DIR})
