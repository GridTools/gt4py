# GridTools CMake coding convention and architecture
# The following guide is intended for CMake developers of the GridTools library,
# users are referred to the GridTools documentation.
# - GridTools cmake modules are in
#   a) /cmake/internal: used exclusively in the GridTools CMake project
#   b) /cmake/public: need to be available in a GridTools installation
# - CMake functions are prefixed with
#   a) gridtools_: functions intended for the user (public API)
#   b) _gt_: private functions, they are not part of the public API
# - GridTools is header-only, which allows for a more flexible CMake installation compared to compiled libraries:
#   the environment from which GridTools was installed is irrelevant for the installation.
#   Therefore, we use the same macro to setup CMake targets in the main CMakeLists.txt (this file) and in the
#   GridToolsConfig.cmake of an installation (see cmake/public/gridtools_setup_target.cmake)
cmake_minimum_required(VERSION 3.18.1)
cmake_policy(VERSION 3.18.1)

file(STRINGS "version.txt" __GT_VERSION)
project(GridTools VERSION ${__GT_VERSION} LANGUAGES CXX)

message(STATUS "Configuring ${PROJECT_NAME} v${__GT_VERSION}" )

unset(__GT_VERSION)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/public" "${CMAKE_CURRENT_LIST_DIR}/cmake/internal")

set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE BOOL "")
mark_as_advanced(CMAKE_EXPORT_NO_PACKAGE_REGISTRY)

# User setting GT_CLANG_CUDA_MODE: decide if Clang-CUDA or NVCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(GT_CLANG_CUDA_MODE "AUTO" CACHE STRING
        "AUTO, Clang-CUDA or NVCC-CUDA; \
        AUTO = Use NVCC if language CUDA is enabled, else prefer Clang-CUDA.")
    set_property(CACHE GT_CLANG_CUDA_MODE PROPERTY STRINGS "AUTO;Clang-CUDA;NVCC-CUDA")
endif()

# User setting GT_INSTALL_EXAMPLES:
# if we are pointing to the default install path (usually system) we will disable installation of examples by default
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(_default_GT_INSTALL_EXAMPLES OFF)
else()
    set(_default_GT_INSTALL_EXAMPLES ON)
endif()
option(GT_INSTALL_EXAMPLES "Install example sources" ${_default_GT_INSTALL_EXAMPLES})
if(GT_INSTALL_EXAMPLES)
    set(GT_INSTALL_EXAMPLES_PATH "${CMAKE_INSTALL_PREFIX}/gridtools_examples" CACHE FILEPATH
        "Specifies where the source codes of examples should be installed")
    mark_as_advanced(CLEAR GT_INSTALL_EXAMPLES_PATH)
else()
    if(GT_INSTALL_EXAMPLES_PATH)
        mark_as_advanced(FORCE GT_INSTALL_EXAMPLES_PATH)
    endif()
endif()

set(REQUIRED_BOOST_VERSION 1.65.1)
find_package(Boost ${REQUIRED_BOOST_VERSION} REQUIRED)

# setup main gridtools target
add_library(gridtools INTERFACE)
add_library(GridTools::gridtools ALIAS gridtools)
target_compile_features(gridtools INTERFACE cxx_std_17)
target_link_libraries(gridtools INTERFACE Boost::boost)
target_compile_definitions(gridtools INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:BOOST_PP_VARIADICS=1>)
target_include_directories(gridtools
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS gridtools EXPORT GridToolsTargets
        INCLUDES DESTINATION include
        )

include(gridtools_setup_targets)
_gt_setup_targets(FALSE "${GT_CLANG_CUDA_MODE}")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # add a target to generate API documentation with Doxygen
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
    endif()

    # Python scripts
    add_subdirectory(pyutils)
endif()

# Enable tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
else()
    # User setting GT_BUILD_TESTING: by default testing is disabled if included via FetchContent/add_subdirectory,
    # GT_BUILD_TESTING=ON will overwrite this default.
    option(GT_BUILD_TESTING "Build GridTools tests" OFF)
    if(GT_BUILD_TESTING)
        include(CTest)
    endif()
     # Note that the following is setting a local variable BUILD_TESTING, not affecting the setting of a super-project
    set(BUILD_TESTING ${GT_BUILD_TESTING})
endif()
if(BUILD_TESTING)
    set(GT_TESTS_CXX_STANDARD "DEFAULT" CACHE STRING "C++ standard for compiling tests (DEFAULT=17).")
    set_property(CACHE GT_TESTS_CXX_STANDARD PROPERTY STRINGS "DEFAULT;20")
    if(NOT GT_TESTS_CXX_STANDARD STREQUAL "DEFAULT")
        set(CMAKE_CXX_STANDARD ${GT_TESTS_CXX_STANDARD})
        set(CMAKE_CUDA_STANDARD ${GT_TESTS_CXX_STANDARD})
    endif()
    if(MPI_CXX_FOUND)
        include(workaround_mpiexec)
        _fix_mpi_exec()
    endif()

    if(TARGET _gridtools_cuda)
        if (DEFINED ENV{CUDA_ARCH})
            set(GT_CUDA_ARCH_INIT $ENV{CUDA_ARCH})
        endif()
        set(GT_CUDA_ARCH "${GT_CUDA_ARCH_INIT}" CACHE STRING "Compute capability for CUDA used for tests")

        # CMake 3.18 support for CMAKE_CUDA_ARCHITECTURES
        if(GT_CUDA_ARCH)
            gt_cuda_arch_to_cuda_arch_version(${GT_CUDA_ARCH} CMAKE_CUDA_ARCHITECTURES)
        else()
            set(CMAKE_CUDA_ARCHITECTURES OFF)
        endif()
    endif()
    include(detect_test_features)
    detect_c_compiler()
    detect_fortran_compiler()
    option(GT_TESTS_REQUIRE_OpenMP "Require OpenMP support." OFF)
    if(GT_TESTS_REQUIRE_OpenMP)
        find_package(OpenMP COMPONENTS CXX REQUIRED)
    endif()
    option(GT_TESTS_REQUIRE_GPU "Require GPU support." OFF)
    if(GT_TESTS_REQUIRE_GPU AND NOT TARGET stencil_gpu)
        message(FATAL_ERROR "GPU targets where required but we couldn't configure GPU support. Please check the CMake error logs.")
    endif()

    add_subdirectory(tests)
    add_subdirectory(docs_src/manuals/getting_started)
endif()

# examples
if(GT_INSTALL_EXAMPLES)
    add_subdirectory(examples)
endif()

# absolute path to avoid collision with "export" from other packages
include(${CMAKE_CURRENT_LIST_DIR}/cmake/internal/export.cmake)

_gt_print_configuration_summary()
