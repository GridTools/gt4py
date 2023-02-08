cmake_minimum_required(VERSION 3.18.1)

project(cpp_backend_tests_driver LANGUAGES CXX)

set(BACKEND NOTFOUND CACHE STRING "fn backend")
set_property(CACHE BACKEND PROPERTY STRINGS "NOTFOUND;naive;gpu")

if(NOT BACKEND)
    message(FATAL_ERROR "No backend selected")
else()
    message(STATUS "Testing backend \"${BACKEND}\"")
endif()

if(BACKEND STREQUAL "gpu")
    enable_language(CUDA)
    set(is_gpu ON)
endif() 
string(TOUPPER ${BACKEND} backend_upper_case)

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(GridTools)

function(generate_computation)
    set(options)
    set(oneValueArgs NAME SRC_FILE GENERATED_FILENAME IMPERATIVE)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_custom_command(OUTPUT ${ARG_GENERATED_FILENAME}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC_FILE} ${ARG_GENERATED_FILENAME} ${ARG_IMPERATIVE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INPUT})
    add_custom_target(generated_${ARG_NAME} DEPENDS ${ARG_GENERATED_FILENAME})
endfunction()
    
add_library(regression_main ${gridtools_SOURCE_DIR}/tests/src/regression_main.cpp)
target_include_directories(regression_main PUBLIC ${gridtools_SOURCE_DIR}/tests/include)
target_link_libraries(regression_main PUBLIC gtest gmock gridtools)

function(add_fn_codegen_test)
    set(options)
    set(oneValueArgs NAME SRC_FILE DRIVER_FILE IMPERATIVE)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(STENCIL_IMPL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/generated_${ARG_NAME}.hpp)
    generate_computation(NAME ${ARG_NAME} SRC_FILE ${ARG_SRC_FILE} GENERATED_FILENAME ${STENCIL_IMPL_SOURCE} IMPERATIVE ${ARG_IMPERATIVE})
    add_executable(${ARG_NAME} ${ARG_DRIVER_FILE})
    target_link_libraries(${ARG_NAME} fn_${BACKEND} regression_main)
    target_compile_definitions(${ARG_NAME} PRIVATE GT_FN_${backend_upper_case})
    target_compile_definitions(${ARG_NAME} PRIVATE GT_FN_BACKEND=${BACKEND})
    target_compile_definitions(${ARG_NAME} PRIVATE GENERATED_FILE=\"${STENCIL_IMPL_SOURCE}\")
    if(is_gpu)
        gridtools_setup_target(${ARG_NAME} CUDA_ARCH sm_75) #TODO
    endif()
    add_dependencies(${ARG_NAME} generated_${ARG_NAME})

    add_test(NAME ${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_NAME}>)
endfunction()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    include(cmake/FetchGoogleTest.cmake)
    fetch_googletest()
    
    add_fn_codegen_test(NAME copy_stencil SRC_FILE copy_stencil.py DRIVER_FILE copy_stencil_driver.cpp)
    add_fn_codegen_test(NAME copy_stencil_field_view SRC_FILE copy_stencil_field_view.py DRIVER_FILE copy_stencil_field_view_driver.cpp)
    add_fn_codegen_test(NAME anton_lap SRC_FILE anton_lap.py DRIVER_FILE anton_lap_driver.cpp)
    add_fn_codegen_test(NAME fvm_nabla_fun SRC_FILE fvm_nabla.py DRIVER_FILE fvm_nabla_driver.cpp IMPERATIVE TRUE)
    add_fn_codegen_test(NAME fvm_nabla_imp SRC_FILE fvm_nabla.py DRIVER_FILE fvm_nabla_driver.cpp IMPERATIVE FALSE)
    add_fn_codegen_test(NAME tridiagonal_solve SRC_FILE tridiagonal_solve.py DRIVER_FILE tridiagonal_solve_driver.cpp)
endif()
    
    
