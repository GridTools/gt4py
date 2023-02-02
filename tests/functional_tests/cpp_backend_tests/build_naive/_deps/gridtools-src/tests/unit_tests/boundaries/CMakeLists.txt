if(TARGET boundaries_gpu)
    gridtools_add_unit_test(test_apply_gpu_internals SOURCES test_apply_gpu_internals.cu LIBRARIES boundaries_gpu LABELS gpu)
    gridtools_add_unit_test(test_boundary_conditions_gpu SOURCES test_boundary_conditions.cpp LIBRARIES boundaries_gpu LABELS gpu)
    target_compile_definitions(test_boundary_conditions_gpu PRIVATE GT_STORAGE_GPU GT_GCL_GPU)
endif()

if(TARGET boundaries_cpu)
    gridtools_add_unit_test(test_boundary_conditions_cpu SOURCES test_boundary_conditions.cpp LIBRARIES boundaries_cpu NO_NVCC)
    target_compile_definitions(test_boundary_conditions_cpu PRIVATE GT_STORAGE_CPU_KFIRST GT_GCL_CPU)
endif()

gridtools_add_unit_test(test_bindbc_utilities SOURCES test_bindbc_utilities.cpp)

if (TARGET gcl_cpu)
    gridtools_add_mpi_test(cpu test_distributed_boundaries_cpu SOURCES test_distributed_boundaries.cpp)
    target_compile_definitions(test_distributed_boundaries_cpu PRIVATE GT_STORAGE_CPU_KFIRST GT_GCL_CPU GT_TIMER_OMP)
endif()

if (TARGET gcl_gpu)
    gridtools_add_mpi_test(gpu test_distributed_boundaries_gpu SOURCES test_distributed_boundaries.cpp)
    target_compile_definitions(test_distributed_boundaries_gpu PRIVATE GT_STORAGE_GPU GT_GCL_GPU GT_TIMER_CUDA)
endif()
