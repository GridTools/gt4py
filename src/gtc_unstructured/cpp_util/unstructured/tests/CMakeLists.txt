include(FetchGoogleTest)
fetch_googletest()

add_library(test_util INTERFACE)
target_include_directories(test_util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(regression)
add_subdirectory(unit_tests)
