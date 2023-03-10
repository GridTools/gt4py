#[=======================================================================[.rst:

GridToolsConfig
---------------

In case the compiler is Clang targeting CUDA, set ``GT_CLANG_CUDA_MODE`` to
``AUTO`` (default), ``Clang-CUDA`` or ``NVCC-CUDA``. ``AUTO`` will use
``Clang-CUDA`` if available.

Targets
^^^^^^^^^^^^^^^^

Depending on the available dependencies (OpenMP, MPI, CUDA) a set of targets is
exported. A configuration summary will be printed in case of successfully
detecting GridTools.

#]=======================================================================]

set(GridTools_VERSION 2.3.0)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was GridToolsConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../../../../../../usr/local" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

get_filename_component(GRIDTOOLS_CONFIG_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Only setup targets, if GridTools was not already included with a add_subdirectory/FetchContent
if(NOT GridTools_BINARY_DIR AND NOT TARGET GridTools::gridtools)
    include(CMakeFindDependencyMacro)
    find_dependency(Boost 1.65.1)

    include("${GRIDTOOLS_CONFIG_CMAKE_DIR}/GridToolsTargets.cmake" )

    list(APPEND CMAKE_MODULE_PATH /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-build/CMakeFiles/build-install/lib/cmake)
    include(gridtools_setup_targets)

    if(NOT DEFINED GT_CLANG_CUDA_MODE)
        set(GT_CLANG_CUDA_MODE AUTO)
    endif()
    _gt_setup_targets(TRUE ${GT_CLANG_CUDA_MODE})
else()
    message(WARNING "find_package(GridTools) ignored, targets are already available.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GridTools REQUIRED_VARS GridTools_VERSION VERSION_VAR GridTools_VERSION)
if(GridTools_FOUND)
    find_package_message(GridTools "  at ${GRIDTOOLS_CONFIG_CMAKE_DIR}" "[${GRIDTOOLS_CONFIG_CMAKE_DIR}]")
    _gt_print_configuration_summary()
endif()
