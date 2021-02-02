cmake_minimum_required(VERSION 3.14.5)
project(unstructured_interface LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/private")

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall)
#add_compile_options(-Wall -Wextra -pedantic -Wno-sign-compare)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # set(CMAKE_CUDA_STANDARD 17)
endif()

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG        master
    # URL https://github.com/GridTools/gridtools/archive/???.tar.gz
)
FetchContent_MakeAvailable(GridTools)

add_library(gtnext INTERFACE)
target_include_directories(gtnext INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(prototype prototype.cc)
target_link_libraries(prototype GridTools::gridtools)
target_include_directories(prototype PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# find_package(eckit REQUIRED)
find_package(Atlas REQUIRED PATHS $ENV{ATLAS_INSTALL_DIR})

add_executable(atlas_mesh_adapter atlas_mesh_adapter.cc)
target_link_libraries(atlas_mesh_adapter atlas GridTools::gridtools)
target_include_directories(atlas_mesh_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(atlas_field_converter atlas_field_converter.cc)
target_link_libraries(atlas_field_converter atlas GridTools::gridtools)
target_include_directories(atlas_field_converter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CUDA_COMPILER)
  add_executable(atlas_field_converter_cuda atlas_field_converter.cu)
  target_link_libraries(atlas_field_converter_cuda atlas GridTools::storage_gpu)
  target_include_directories(atlas_field_converter_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(atlas_field_converter_cuda PRIVATE "-std=c++17")
  target_compile_options(atlas_field_converter_cuda PRIVATE "-arch=sm_50")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
