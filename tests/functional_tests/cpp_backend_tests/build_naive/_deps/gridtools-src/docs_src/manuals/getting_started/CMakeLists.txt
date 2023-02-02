
function(gridtools_add_getting_started_test tgt_name)
    set(options)
    set(one_value_args)
    set(multi_value_args LIBRARIES SOURCES LABELS)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    set(tgt getting_started_${tgt_name})
    add_executable(${tgt} ${ARGS_SOURCES})
    target_link_libraries(${tgt} PRIVATE gridtools ${ARGS_LIBRARIES})
    gridtools_setup_target(${tgt} CUDA_ARCH ${GT_CUDA_ARCH})
    add_test(NAME ${tgt} COMMAND $<TARGET_FILE:${tgt}>)
    set(labels getting_started ${ARGS_LABELS})
    set_tests_properties(${tgt} PROPERTIES LABELS "${labels}")
endfunction()

gridtools_add_getting_started_test(test_naive_implementation SOURCES code/test_naive_implementation.cpp)
gridtools_add_getting_started_test(test_gt_storage SOURCES code/test_gt_storage.cpp)

if(TARGET storage_gpu)
    gridtools_add_getting_started_test(test_gt_storage_gpu
            SOURCES code/test_gt_storage.cpp
            LIBRARIES storage_gpu
            LABELS gpu)
endif()

if(TARGET stencil_cpu_ifirst)
    gridtools_add_getting_started_test(test_gt_laplacian
            SOURCES code/test_gt_laplacian.cpp LIBRARIES stencil_cpu_ifirst)
    gridtools_add_getting_started_test(test_gt_smoothing_variant1
            SOURCES code/test_gt_smoothing_variant1.cpp LIBRARIES stencil_cpu_ifirst)
    gridtools_add_getting_started_test(test_gt_smoothing_variant2
            SOURCES code/test_gt_smoothing_variant2.cpp LIBRARIES stencil_cpu_ifirst)
    gridtools_add_getting_started_test(test_gt_smoothing_variant3
            SOURCES code/test_gt_smoothing_variant3.cpp LIBRARIES stencil_cpu_ifirst)
endif()

if(TARGET stencil_gpu)
    gridtools_add_getting_started_test(test_gt_laplacian_gpu
            SOURCES code/test_gt_laplacian.cpp LIBRARIES stencil_gpu LABELS gpu)
    gridtools_add_getting_started_test(test_gt_smoothing_variant1_gpu
            SOURCES code/test_gt_smoothing_variant1.cpp LIBRARIES stencil_gpu LABELS gpu)
    gridtools_add_getting_started_test(test_gt_smoothing_variant2_gpu
            SOURCES code/test_gt_smoothing_variant2.cpp LIBRARIES stencil_gpu LABELS gpu)
    gridtools_add_getting_started_test(test_gt_smoothing_variant3_gpu
            SOURCES code/test_gt_smoothing_variant3.cpp LIBRARIES stencil_gpu LABELS gpu)
endif()

# Test the CMake example file
# The generated file should be part of the repository to be able to build the docs without having to run CMake.
configure_file(code/CMakeLists.txt.in ${CMAKE_CURRENT_LIST_DIR}/code/CMakeLists.txt @ONLY)

include(ExternalProject)
ExternalProject_Add(getting_started_cmake_test
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/code
    CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DGridTools_DIR=${PROJECT_BINARY_DIR} # relies on the GridToolsConfig.cmake in the build directory (the one used for export(package))
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    INSTALL_COMMAND ""
    )
