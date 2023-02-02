gridtools_check_compilation(test_integral_constant test_integral_constant.cpp)
gridtools_check_compilation(test_layout_map test_layout_map.cpp)

gridtools_add_unit_test(test_array SOURCES test_array.cpp)
gridtools_add_unit_test(test_compose SOURCES test_compose.cpp)
gridtools_add_unit_test(test_hugepage_alloc SOURCES test_hugepage_alloc.cpp)
gridtools_add_unit_test(test_hymap SOURCES test_hymap.cpp)
gridtools_add_unit_test(test_pair SOURCES test_pair.cpp)
gridtools_add_unit_test(test_stride_util SOURCES test_stride_util.cpp)
gridtools_add_unit_test(test_tuple_util SOURCES test_tuple_util.cpp)
gridtools_add_unit_test(test_for_each SOURCES test_for_each.cpp)
gridtools_add_unit_test(test_ct_dispatch SOURCES test_ct_dispatch.cpp)

gridtools_add_unit_test(test_atomic_functions SOURCES test_atomic_functions.cpp NO_NVCC)
gridtools_add_unit_test(test_cuda_is_ptr SOURCES test_cuda_is_ptr.cpp NO_NVCC)
gridtools_add_unit_test(test_gt_math SOURCES test_gt_math.cpp NO_NVCC)
gridtools_add_unit_test(test_hypercube_iterator SOURCES test_hypercube_iterator.cpp NO_NVCC)
gridtools_add_unit_test(test_tuple SOURCES test_tuple.cpp NO_NVCC)
gridtools_add_unit_test(test_int_vector SOURCES test_int_vector.cpp NO_NVCC)

if(TARGET _gridtools_cuda)
    gridtools_check_compilation(test_cuda_type_traits test_cuda_type_traits.cu)

    gridtools_add_unit_test(test_hymap_cuda
            SOURCES test_hymap.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
    gridtools_add_unit_test(test_atomic_functions_cuda
            SOURCES test_atomic_functions.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
    gridtools_add_unit_test(test_cuda_is_ptr_cuda
            SOURCES test_cuda_is_ptr.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
    gridtools_add_unit_test(test_gt_math_cuda
            SOURCES test_gt_math.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
    gridtools_add_unit_test(test_hypercube_iterator_cuda
            SOURCES test_hypercube_iterator.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
    gridtools_add_unit_test(test_int_vector_cuda
        SOURCES test_int_vector.cu
        LIBRARIES _gridtools_cuda
        LABELS cuda)
    gridtools_add_unit_test(test_tuple_cuda
            SOURCES test_tuple.cu
            LIBRARIES _gridtools_cuda
            LABELS cuda)
endif()
