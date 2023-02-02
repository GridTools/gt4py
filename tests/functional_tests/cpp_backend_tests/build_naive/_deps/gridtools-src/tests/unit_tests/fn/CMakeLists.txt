gridtools_add_unit_test(test_extents SOURCES test_extents.cpp LABELS fn)
gridtools_add_unit_test(test_fn_backend_naive SOURCES test_fn_backend_naive.cpp LABELS fn)
gridtools_add_unit_test(test_fn_cartesian SOURCES test_fn_cartesian.cpp LABELS fn)
gridtools_add_unit_test(test_fn_executor SOURCES test_fn_executor.cpp LABELS fn)
gridtools_add_unit_test(test_fn_neighbor_table SOURCES test_fn_neighbor_table.cpp LABELS fn)
gridtools_add_unit_test(test_fn_run SOURCES test_fn_run.cpp)
gridtools_add_unit_test(test_fn_column_stage SOURCES test_fn_column_stage.cpp)
gridtools_add_unit_test(test_fn_stencil_stage SOURCES test_fn_stencil_stage.cpp LABELS fn)
gridtools_add_unit_test(test_fn_unstructured SOURCES test_fn_unstructured.cpp LABELS fn)

if(TARGET _gridtools_cuda)
    gridtools_add_unit_test(test_fn_backend_gpu_cuda
            SOURCES test_fn_backend_gpu.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda fn)
    gridtools_add_unit_test(test_fn_cartesian_gpu_cuda
            SOURCES test_fn_cartesian_gpu.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda fn)
    gridtools_add_unit_test(test_fn_column_stage_cuda
            SOURCES test_fn_column_stage.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda fn)
    gridtools_add_unit_test(test_fn_unstructured_gpu_cuda
            SOURCES test_fn_unstructured_gpu.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda fn)
    gridtools_add_unit_test(test_extents_cuda
        SOURCES test_extents.cu
        LIBRARIES _gridtools_cuda
        LABELS cuda fn)
endif()
