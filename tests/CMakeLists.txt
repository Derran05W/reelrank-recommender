file(GLOB RR_TEST_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp
)

add_executable(reel_rank_tests ${RR_TEST_SOURCES})
target_link_libraries(reel_rank_tests
    PRIVATE
        rr_core
        vector_db::vdb_core   # needed only by the vdb link-sanity TU
        GTest::gtest_main
        rr_warnings
)
target_compile_definitions(reel_rank_tests PRIVATE RR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(reel_rank_tests)
