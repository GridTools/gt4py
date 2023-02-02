# This test checks that even if the CUDA language is not enabled, we can still use cuda storages,
# as they don't require a CUDA compiler, but just link to cudart.
# Note that this test is meaningless with Clang as it is (almost) always a CUDA compiler.
cmake_minimum_required(VERSION 3.18.1)

project(test_storage_gpu LANGUAGES CXX)

# Disabling C++ extensions is only required for HIP.
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    gridtools
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..
)
FetchContent_MakeAvailable(gridtools)
add_executable(main main.cpp)
target_link_libraries(main GridTools::storage_gpu)
