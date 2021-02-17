function(add_gtnext_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(NOT ARG_SOURCES)
        set(ARG_SOURCES ${ARG_NAME}.cc)
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})
    target_link_libraries(${ARG_NAME} gtnext)
    target_link_libraries(${ARG_NAME} test_util)
    target_link_libraries(${ARG_NAME} GridTools::gridtools gtest gtest_main gmock ${ARG_LIBRARIES})

    add_test(NAME ${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_NAME}>)
endfunction()

function(add_gtnext_atlas_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})


    add_gtnext_test(NAME ${ARG_NAME} SOURCES ${ARG_SOURCES} LIBRARIES atlas eckit)
endfunction()

add_subdirectory(test_helper)

add_gtnext_atlas_test(NAME test_atlas_array_view_adapter)
add_gtnext_atlas_test(NAME test_atlas_field_util SOURCES test_atlas_field_util.cc)
add_gtnext_atlas_test(NAME test_mesh SOURCES test_mesh.cc)
