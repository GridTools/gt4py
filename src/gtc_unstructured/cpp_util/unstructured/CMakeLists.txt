cmake_minimum_required(VERSION 3.14.5)
project(usid LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/private")

add_compile_options(-Wall)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG        master
    # URL https://github.com/GridTools/gridtools/archive/???.tar.gz
)
FetchContent_MakeAvailable(GridTools)

add_library(usid_cpu INTERFACE)
target_include_directories(usid_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(usid_cpu INTERFACE GridTools::gridtools)
target_compile_features(usid_cpu INTERFACE cxx_std_14)
add_library(GridTools::usid_cpu ALIAS usid_cpu)

if(CMAKE_CUDA_COMPILER)
    add_library(usid_gpu INTERFACE)
    target_include_directories(usid_gpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(usid_gpu INTERFACE GridTools::storage_gpu)
    target_compile_features(usid_gpu INTERFACE cxx_std_14)
    target_compile_features(usid_gpu INTERFACE cuda_std_14)
    add_library(GridTools::usid_gpu ALIAS usid_gpu)
endif()

find_package(Atlas PATHS $ENV{ATLAS_INSTALL_DIR})

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
