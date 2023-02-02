if(NOT TARGET stencil_gpu)
    return()
endif()

gridtools_add_unit_test(test_launch_kernel SOURCES test_launch_kernel.cu LIBRARIES stencil_gpu LABELS gpu)
gridtools_add_unit_test(test_shared_allocator SOURCES test_shared_allocator.cu LIBRARIES stencil_gpu LABELS gpu)
gridtools_add_unit_test(test_simple_device_memory_allocator
        SOURCES test_simple_device_memory_allocator.cu LIBRARIES stencil_gpu LABELS gpu)
gridtools_add_unit_test(test_ij_cache SOURCES test_ij_cache.cu LIBRARIES stencil_gpu LABELS gpu)
gridtools_add_unit_test(test_tmp_storage_sid_gpu SOURCES test_tmp_storage_sid.cu LIBRARIES stencil_gpu LABELS gpu)
gridtools_add_unit_test(test_tmp_storage_sid_gpu_host SOURCES test_tmp_storage_sid.cpp NO_NVCC)

