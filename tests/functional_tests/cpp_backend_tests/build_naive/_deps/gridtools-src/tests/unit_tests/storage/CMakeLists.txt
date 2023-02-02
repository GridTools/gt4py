function(gridtools_add_storage_test tgt_name)
    set(options SKIP_GPU)
    set(one_value_args)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    foreach(storage IN LISTS GT_STORAGES)
        if(ARGS_SKIP_GPU AND storage STREQUAL "gpu")
            return()
        endif()
        set(tgt ${tgt_name}_${storage})
        gridtools_add_unit_test(${tgt}
                SOURCES ${ARGS_SOURCES}
                LIBRARIES storage_${storage}
                LABELS storage ${storage}
                NO_NVCC)
        string(TOUPPER ${storage} u_storage)
        target_compile_definitions(${tgt} PRIVATE GT_STORAGE_${u_storage})
    endforeach()
endfunction()

gridtools_add_unit_test(test_storage_info SOURCES test_storage_info.cpp LABELS storage)

gridtools_add_storage_test(test_storage_sid SOURCES test_storage_sid.cpp)
gridtools_add_storage_test(test_storage_facility SOURCES test_storage_facility.cpp SKIP_GPU) # see below
gridtools_add_storage_test(test_alignment_inner_region SOURCES test_alignment_inner_region.cpp)
gridtools_add_storage_test(test_data_store SOURCES test_data_store.cpp)
gridtools_add_storage_test(test_host_view SOURCES test_host_view.cpp)


# tests requiring a CUDA compiler
if(TARGET storage_gpu AND TARGET _gridtools_cuda)
    gridtools_add_unit_test(test_target_view_gpu SOURCES test_target_view.cu LIBRARIES storage_gpu _gridtools_cuda LABELS storage gpu NO_NVCC)

    # for test_storage_facility_cuda we need a CUDA compiler for validation, therefore explicitly linking to _gridtools_cuda
    gridtools_add_unit_test(test_storage_facility_gpu SOURCES test_storage_facility.cpp LIBRARIES storage_gpu _gridtools_cuda LABELS storage gpu)
    target_compile_definitions(test_storage_facility_gpu PRIVATE GT_STORAGE_GPU)
endif()

add_subdirectory(adapter)
