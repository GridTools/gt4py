function(gridtools_check_compilation tgt)
    add_library(${tgt} OBJECT ${ARGN})
    target_link_libraries(${tgt} gridtools)
    if(TARGET _gridtools_nvcc)
        _gt_convert_to_cuda_source(sources ${ARGN})
        if (NOT sources STREQUAL ARGN)
            add_library(${tgt}_nvcc OBJECT ${sources})
            target_link_libraries(${tgt}_nvcc gridtools)
        endif()
    endif()
endfunction()

function(gridtools_add_unit_test tgt)
    set(options NO_NVCC)
    set(one_value_args)
    set(multi_value_args LIBRARIES SOURCES LABELS)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    gridtools_add_test_executable(${tgt}
            SOURCES ${ARGS_SOURCES}
            LIBRARIES gtest_main gmock ${ARGS_LIBRARIES})

    _gt_depends_on_nvcc(nvcc_compilation_checked ${tgt})

    if(NOT ARGS_NO_NVCC AND TARGET _gridtools_nvcc AND NOT nvcc_compilation_checked)
        # Additional nvcc target is created here. It exists for checking nvvc compilation only.
        # Note that it is an `OBJECT` library.
        # _gridtools_nvcc is added to deps to pickup cuda specific compilation flags.
        _gt_convert_to_cuda_source(sources ${ARGS_SOURCES})
        add_library(${tgt}_nvcc OBJECT ${sources})
        target_link_libraries(${tgt}_nvcc GridToolsTest gtest_main gmock _gridtools_nvcc ${ARGS_LIBRARIES})
    endif()

    add_test(NAME ${tgt} COMMAND $<TARGET_FILE:${tgt}>)
    set(labels unit_test ${ARGS_LABELS})
    set_tests_properties(${tgt} PROPERTIES LABELS "${labels}")
endfunction()

gridtools_check_compilation(test_meta test_meta.cpp)

add_subdirectory(common)
add_subdirectory(sid)
add_subdirectory(boundaries)
add_subdirectory(stencil)
add_subdirectory(storage)
add_subdirectory(layout_transformation)
add_subdirectory(fn)
