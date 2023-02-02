if (NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_Fortran_COMPILER_LOADED)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
        cpp_bindgen
        GIT_REPOSITORY https://github.com/GridTools/cpp_bindgen.git
        GIT_TAG        v1.0.1
)

set(build_testing_ ${BUILD_TESTING})
set(BUILD_TESTING OFF)
FetchContent_GetProperties(cpp_bindgen)
if(NOT cpp_bindgen_POPULATED)
  FetchContent_Populate(cpp_bindgen)
  add_subdirectory(${cpp_bindgen_SOURCE_DIR} ${cpp_bindgen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
set(BUILD_TESTING ${build_testing_})

bindgen_add_library(implementation SOURCES implementation.cpp FORTRAN_MODULE_NAME implementation)
target_link_libraries(implementation PRIVATE gridtools)

if (CMAKE_C_COMPILER_LOADED)
    add_executable(driver driver.c)
    target_link_libraries(driver implementation_c)
endif()

if (CMAKE_Fortran_COMPILER_LOADED)
    bindgen_enable_fortran_library(implementation)
    add_executable(fdriver fdriver.f90)
    target_link_libraries(fdriver implementation_fortran)
    set_target_properties(fdriver PROPERTIES LINKER_LANGUAGE Fortran)

    bindgen_add_library(implementation_wrapper
            SOURCES implementation_wrapper.cpp FORTRAN_MODULE_NAME implementation_wrapper)
    target_link_libraries(implementation_wrapper PRIVATE gridtools)
    bindgen_enable_fortran_library(implementation_wrapper)

    add_executable(fdriver_wrapper fdriver_wrapper.f90)
    target_link_libraries(fdriver_wrapper implementation_wrapper_fortran)
    set_target_properties(fdriver_wrapper PROPERTIES LINKER_LANGUAGE Fortran)
endif()
