if(TARGET _gridtools_cuda)
    gridtools_add_unit_test(test_layout_transformation_gpu
            SOURCES test_layout_transformation.cpp
            LIBRARIES _gridtools_cuda
            LABELS gpu)
endif()

if(OpenMP_CXX_FOUND)
    gridtools_add_unit_test(test_layout_transformation
            SOURCES test_layout_transformation.cpp
            LIBRARIES OpenMP::OpenMP_CXX
            NO_NVCC)
else()
    gridtools_add_unit_test(test_layout_transformation
            SOURCES test_layout_transformation.cpp
            NO_NVCC)
endif()
