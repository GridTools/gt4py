
if (OpenMP_CXX_FOUND)
    set(omp_ OpenMP::OpenMP_CXX)
endif()
gridtools_add_test_library(regression_main SOURCES regression_main.cpp LIBRARIES PRIVATE ${omp_})
unset(omp_)

if (TARGET gcl_cpu)
    add_library(mpi_gtest_main_cpu STATIC EXCLUDE_FROM_ALL mpi_test_driver.cpp)
    target_link_libraries(mpi_gtest_main_cpu gridtools gtest MPI::MPI_CXX)
endif()

if (TARGET gcl_gpu)
    add_library(mpi_gtest_main_gpu STATIC EXCLUDE_FROM_ALL mpi_test_driver.cpp)
    target_compile_definitions(mpi_gtest_main_gpu PRIVATE GT_HAS_CUDA)
    target_link_libraries(mpi_gtest_main_gpu PUBLIC gridtools gtest MPI::MPI_CXX)
    if(TARGET CUDA::cudart) # doesn't exist if HIP mode
        target_link_libraries(mpi_gtest_main_gpu PUBLIC CUDA::cudart)
    endif()
endif()
