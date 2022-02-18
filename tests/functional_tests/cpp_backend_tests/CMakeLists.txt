cmake_minimum_required(VERSION 3.16.2)

project(cpp_backend_tests_driver LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/fthaler/gridtools.git
    GIT_TAG        fn-frontend
)
FetchContent_MakeAvailable(GridTools)

function(generate_computation)
    set(options)
    set(oneValueArgs NAME SRC_FILE)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(STENCIL_IMPL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/generated_${ARG_NAME}.hpp)
    add_custom_command(OUTPUT ${STENCIL_IMPL_SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC_FILE} ${STENCIL_IMPL_SOURCE} 
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INPUT})
    add_custom_target(generated_${ARG_NAME} DEPENDS ${STENCIL_IMPL_SOURCE})
endfunction()
    
add_library(regression_main ${gridtools_SOURCE_DIR}/tests/src/regression_main.cpp)
target_include_directories(regression_main PUBLIC ${gridtools_SOURCE_DIR}/tests/include)
target_link_libraries(regression_main PUBLIC gtest gmock gridtools)

function(add_fn_codegen_test)
    set(options)
    set(oneValueArgs NAME SRC_FILE DRIVER_FILE)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    generate_computation(NAME ${ARG_NAME} SRC_FILE ${ARG_SRC_FILE})
    add_executable(${ARG_NAME} ${ARG_DRIVER_FILE})
    target_link_libraries(${ARG_NAME} fn_naive regression_main)
    target_compile_definitions(${ARG_NAME} PRIVATE GT_FN_NAIVE)

    add_test(NAME ${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_NAME}>)
endfunction()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    include(cmake/FetchGoogleTest.cmake)
    fetch_googletest()
    
    # generate_computation(NAME copy_stencil SRC_FILE copy_stencil.py)
    add_fn_codegen_test(NAME copy_stencil SRC_FILE copy_stencil.py DRIVER_FILE copy_stencil_driver.cpp)
endif()
    
    
