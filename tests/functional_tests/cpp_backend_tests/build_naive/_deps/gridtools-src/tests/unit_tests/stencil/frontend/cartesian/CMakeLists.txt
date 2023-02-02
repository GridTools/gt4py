function(gridtools_add_cartesian_test tgt_name)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    foreach(backend IN LISTS GT_STENCILS)
        set(tgt ${tgt_name}_${backend})
        gridtools_add_unit_test(${tgt}
                SOURCES ${ARGS_SOURCES}
                LIBRARIES stencil_${backend}
                LABELS cartesian ${backend}
                NO_NVCC)
        string(TOUPPER ${backend} u_backend)
        target_compile_definitions(${tgt} PRIVATE GT_STENCIL_${u_backend})
    endforeach()
endfunction()

gridtools_check_compilation(test_call_stress_types test_call_stress_types.cpp)
gridtools_check_compilation(test_call_proc_stress_types test_call_proc_stress_types.cpp)
gridtools_check_compilation(test_arg_extent_intent test_arg_extent_intent.cpp)
gridtools_check_compilation(test_stage_with_extents test_stage_with_extents.cpp)

gridtools_add_unit_test(test_accessor SOURCES test_accessor.cpp)
gridtools_add_unit_test(test_call_interfaces SOURCES test_call_interfaces.cpp)
gridtools_add_unit_test(test_call_proc_interfaces SOURCES test_call_proc_interfaces.cpp)
gridtools_add_unit_test(test_expandable_parameters SOURCES test_expandable_parameters.cpp)
gridtools_add_unit_test(test_expressions_integration SOURCES test_expressions_integration.cpp)
gridtools_add_unit_test(test_multi_types SOURCES test_multi_types.cpp)
gridtools_add_unit_test(test_stencils SOURCES test_stencils.cpp)

gridtools_add_cartesian_test(test_kcache_fill SOURCES test_kcache_fill.cpp)
gridtools_add_cartesian_test(test_kcache_fill_and_flush SOURCES test_kcache_fill_and_flush.cpp)
gridtools_add_cartesian_test(test_kcache_flush SOURCES test_kcache_flush.cpp)
gridtools_add_cartesian_test(test_kcache_local SOURCES test_kcache_local.cpp)
gridtools_add_cartesian_test(test_kparallel SOURCES test_kparallel.cpp)

gridtools_add_unit_test(test_expressions SOURCES test_expressions.cpp NO_NVCC)

if(TARGET _gridtools_cuda)
    gridtools_add_unit_test(test_expressions_cuda SOURCES test_expressions.cu LIBRARIES _gridtools_cuda LABELS cuda)
endif()
