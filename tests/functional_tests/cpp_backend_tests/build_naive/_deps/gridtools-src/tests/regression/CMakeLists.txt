add_executable(perftests)

file(WRITE 
     "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp"
     "// This file is needed in some cases because CMake targets must have at least one .cpp source."
)
target_sources(perftests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
if (TARGET _gridtools_cuda)
    target_link_libraries(perftests gridtools _gridtools_cuda)
endif()
gridtools_set_gpu_arch_on_target(perftests "${GT_CUDA_ARCH}")

target_link_libraries(perftests regression_main)

if(TARGET threadpool_hpx)
    # These fake targets should not be used by the user, they are just to parametrize the tests on the threadpool
    list(APPEND GT_STENCILS cpu_kfirst_hpx cpu_ifirst_hpx)

    add_library(stencil_cpu_kfirst_hpx INTERFACE)
    target_link_libraries(stencil_cpu_kfirst_hpx INTERFACE stencil_cpu_kfirst threadpool_hpx)

    add_library(stencil_cpu_ifirst_hpx INTERFACE)
    target_link_libraries(stencil_cpu_ifirst_hpx INTERFACE stencil_cpu_ifirst threadpool_hpx)
endif()

function(gridtools_add_regression_test tgt_name)
    set(options PERFTEST)
    set(one_value_args LIB_PREFIX)
    set(multi_value_args KEYS SOURCES LABELS)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    foreach(key IN LISTS ARGS_KEYS)
        set(tgt ${tgt_name}_${key})
        add_library(${tgt}_lib OBJECT)
        gridtools_setup_test_target(${tgt}_lib
                SOURCES ${ARGS_SOURCES}
                LIBRARIES gridtools regression_main ${ARGS_LIB_PREFIX}_${key})
        add_executable(${tgt} $<TARGET_OBJECTS:${tgt}_lib>)
        target_link_libraries(${tgt} regression_main ${ARGS_LIB_PREFIX}_${key})
        add_test(NAME ${tgt} COMMAND $<TARGET_FILE:${tgt}>)
        set(labels regression ${key} ${ARGS_LABELS})
        set_tests_properties(${tgt} PROPERTIES LABELS "${labels}")
        if (ARGS_PERFTEST AND NOT key STREQUAL naive)
            set_property(TARGET perftests APPEND PROPERTY SOURCES $<TARGET_OBJECTS:${tgt}_lib>)
            target_link_libraries(perftests ${ARGS_LIB_PREFIX}_${key})
        endif()
    endforeach()
endfunction()

function(add_backend_testees prefix)
    foreach(backend IN LISTS ARGN)
        set(tgt ${prefix}_${backend})
        add_library(${tgt} INTERFACE)
        target_link_libraries(${tgt} INTERFACE stencil_${backend})
        string(TOUPPER ${backend} u_backend)
        target_compile_definitions(${tgt} INTERFACE GT_STENCIL_${u_backend})
    endforeach()
endfunction()
add_backend_testees(backend_testee ${GT_STENCILS})

function(gridtools_add_cartesian_regression_test tgt_name)
    gridtools_add_regression_test(${tgt_name} ${ARGN}
            LIB_PREFIX backend_testee
            KEYS ${GT_STENCILS}
            LABELS cartesian)
endfunction()

function(add_reduction_testees prefix)
    foreach(backend IN LISTS ARGN)
        set(tgt ${prefix}_${backend})
        add_library(${tgt} INTERFACE)
        target_link_libraries(${tgt} INTERFACE reduction_${backend})
        string(TOUPPER ${backend} u_backend)
        target_compile_definitions(${tgt} INTERFACE GT_REDUCTION_${u_backend})
        if (backend STREQUAL gpu)
            target_link_libraries(${tgt} INTERFACE stencil_gpu storage_gpu)
        elseif (backend STREQUAL cpu)
            target_link_libraries(${tgt} INTERFACE stencil_cpu_ifirst storage_cpu_ifirst)
        elseif (backend STREQUAL naive)
            target_link_libraries(${tgt} INTERFACE stencil_naive storage_cpu_kfirst)
        endif()
    endforeach()
endfunction()
add_reduction_testees(reduction_testee ${GT_REDUCTIONS})

function(gridtools_add_reduction_test tgt_name)
    gridtools_add_regression_test(${tgt_name} ${ARGN}
            LIB_PREFIX reduction_testee
            KEYS ${GT_REDUCTIONS}
            LABELS reduction)
endfunction()

function(gridtools_add_layout_transformation_test)
    foreach(arch IN LISTS GT_GCL_ARCHS)
        set(tgt layout_transformation_testee_${arch})
        add_library(${tgt} INTERFACE)
        target_link_libraries(${tgt} INTERFACE layout_transformation_${arch})
        if (arch STREQUAL gpu)
            target_compile_definitions(${tgt} INTERFACE GT_STORAGE_GPU GT_TIMER_CUDA)
        elseif (arch STREQUAL cpu)
            target_compile_definitions(${tgt} INTERFACE GT_STORAGE_CPU_KFIRST GT_TIMER_OMP)
        endif()
    endforeach()
    gridtools_add_regression_test(layout_transformation_test
            LIB_PREFIX layout_transformation_testee
            KEYS ${GT_GCL_ARCHS}
            SOURCES layout_transformation.cpp
            PERFTEST)
endfunction()

function(gridtools_add_boundary_conditions_test)
    set(ENABLED_GT_GCL_ARCHS)
    foreach(arch IN LISTS GT_GCL_ARCHS)
        if(arch STREQUAL gpu AND CMAKE_BUILD_TYPE STREQUAL Debug AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1)
            message(WARNING "CUDA 11.0.x is buggy, see https://github.com/GridTools/gridtools/issues/1522, skipping test boundary_conditions_gpu.")
        else()
            list(APPEND ENABLED_GT_GCL_ARCHS ${arch})
            set(tgt bc_testee_${arch})
            add_library(${tgt} INTERFACE)
            target_link_libraries(${tgt} INTERFACE boundaries_${arch})
            if (arch STREQUAL gpu)
                target_compile_definitions(${tgt} INTERFACE GT_GCL_GPU)
            elseif (arch STREQUAL cpu)
                target_compile_definitions(${tgt} INTERFACE GT_GCL_CPU)
            endif()
        endif()
    endforeach()
    gridtools_add_regression_test(boundary_conditions
            LIB_PREFIX bc_testee
            KEYS ${ENABLED_GT_GCL_ARCHS}
            SOURCES boundary_conditions.cpp
            PERFTEST)
endfunction()

if (TARGET gcl_cpu AND TARGET stencil_cpu_kfirst)
    gridtools_add_mpi_test(cpu copy_stencil_parallel_cpu SOURCES copy_stencil_parallel.cpp LIBRARIES stencil_cpu_kfirst)
    target_compile_definitions(copy_stencil_parallel_cpu PRIVATE GT_STENCIL_CPU_KFIRST GT_GCL_CPU)
endif()

if (TARGET gcl_gpu AND TARGET stencil_gpu)
    gridtools_add_mpi_test(gpu copy_stencil_parallel_gpu SOURCES copy_stencil_parallel.cpp LIBRARIES stencil_gpu)
    target_compile_definitions(copy_stencil_parallel_gpu PRIVATE GT_STENCIL_GPU GT_GCL_GPU)
endif()

gridtools_add_cartesian_regression_test(horizontal_diffusion SOURCES horizontal_diffusion.cpp PERFTEST)
gridtools_add_cartesian_regression_test(horizontal_diffusion_fused SOURCES horizontal_diffusion_fused.cpp PERFTEST)
gridtools_add_cartesian_regression_test(simple_hori_diff SOURCES simple_hori_diff.cpp PERFTEST)
gridtools_add_cartesian_regression_test(copy_stencil SOURCES copy_stencil.cpp PERFTEST)
gridtools_add_cartesian_regression_test(vertical_advection_dycore SOURCES vertical_advection_dycore.cpp PERFTEST)
gridtools_add_cartesian_regression_test(advection_pdbott_prepare_tracers SOURCES advection_pdbott_prepare_tracers.cpp PERFTEST)
gridtools_add_cartesian_regression_test(parallel_multistage_fusion SOURCES parallel_multistage_fusion.cpp)
gridtools_add_cartesian_regression_test(laplacian SOURCES laplacian.cpp)
gridtools_add_cartesian_regression_test(positional_stencil SOURCES positional_stencil.cpp)
gridtools_add_cartesian_regression_test(tridiagonal SOURCES tridiagonal.cpp)
gridtools_add_cartesian_regression_test(alignment SOURCES alignment.cpp)
gridtools_add_cartesian_regression_test(extended_4D SOURCES extended_4D.cpp)
gridtools_add_cartesian_regression_test(expandable_parameters SOURCES expandable_parameters.cpp)
gridtools_add_cartesian_regression_test(expandable_parameters_single_kernel SOURCES expandable_parameters_single_kernel.cpp)
gridtools_add_cartesian_regression_test(horizontal_diffusion_functions SOURCES horizontal_diffusion_functions.cpp)
gridtools_add_cartesian_regression_test(whole_axis_access SOURCES whole_axis_access.cpp)
gridtools_add_reduction_test(scalar_product SOURCES scalar_product.cpp PERFTEST)
gridtools_add_layout_transformation_test()
gridtools_add_boundary_conditions_test()

add_executable(c_array_copy c_array_copy.cpp)
target_link_libraries(c_array_copy gtest_main gmock gridtools)
add_test(NAME c_array_copy COMMAND $<TARGET_FILE:c_array_copy>)

if(TARGET stencil_dump)
    add_executable(dump dump.cpp)
    target_link_libraries(dump gtest_main gmock gridtools stencil_dump)
endif()

add_subdirectory(icosahedral)
add_subdirectory(c_bindings)
add_subdirectory(py_bindings)
add_subdirectory(gcl)
add_subdirectory(fn)
