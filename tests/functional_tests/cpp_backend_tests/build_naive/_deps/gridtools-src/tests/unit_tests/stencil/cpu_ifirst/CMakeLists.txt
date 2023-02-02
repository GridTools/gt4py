if(NOT TARGET stencil_cpu_ifirst)
    return()
endif()

gridtools_add_unit_test(test_tmp_storage_sid_cpu_ifirst SOURCES test_tmp_storage_sid.cpp LIBRARIES stencil_cpu_ifirst NO_NVCC)
