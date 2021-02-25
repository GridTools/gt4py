cmake_minimum_required(VERSION 3.14.5)
project(gtc_regression LANGUAGES CXX)

cmake_policy(SET CMP0094 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/private")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_EXTENSIONS OFF)

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

FetchContent_Declare(cpputil_usid
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/gtc_unstructured/cpp_util
)
FetchContent_MakeAvailable(cpputil_usid)


function(add_regression_test_executable)
    set(options)
    set(oneValueArgs NAME INPUT MODE)
    set(multiValueArgs SOURCES LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "${ARG_NAME}")

    # generate c++ from gtscript
    set(STENCIL_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(STENCIL_IMPL_SOURCE ${STENCIL_OUTPUT_DIR}/generated_${ARG_NAME}.hpp)
    set(BACKEND ${ARG_MODE})
    add_custom_command(OUTPUT ${STENCIL_IMPL_SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generator.py ${ARG_INPUT} ${STENCIL_OUTPUT_DIR} ${ARG_MODE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INPUT}.py)
    add_custom_target(generated_${ARG_NAME} DEPENDS ${STENCIL_IMPL_SOURCE})


    # generate driver code
    #  create directory to generate code to
    set(gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated_${ARG_NAME})
    file(MAKE_DIRECTORY ${gen_dir})
    #  the gridtools_setup_target expect a relative path so transform it
    file(RELATIVE_PATH bin_dir_rel_to_src_dir ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/.)
    set(generated_arg_sources)
    foreach(arg_source ${ARG_SOURCES})
        get_filename_component(generated_source ${arg_source} NAME)
        configure_file(${arg_source} ${gen_dir}/${generated_source})
        list(APPEND generated_arg_sources ${bin_dir_rel_to_src_dir}/generated_${ARG_NAME}/${generated_source})
    endforeach()

    # setup actual test
    add_executable(${ARG_NAME} ${generated_arg_sources}) # TODO fix GridTools appending .cu to .hpp, then we don't need the custom_target
    target_link_libraries(${ARG_NAME} ${ARG_LIBRARIES})
    add_dependencies(${ARG_NAME} generated_${ARG_NAME})
    add_test(NAME ${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_NAME}>)
endfunction()

function(add_regression_test name)
    set(options)
    set(oneValueArgs DRIVER)
    set(multiValueArgs ADDITIONAL_LIBRARIES)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "${options}" "${oneValueArgs}" "${multiValueArgs}")

    if(ARG_DRIVER)
        set(_driver ${ARG_DRIVER})
    else()
        set(_driver ${name})
    endif()
    set(_sources ${_driver}_driver.cc)
    set(libraries GridTools::gridtools gtest gtest_main gmock)
    list (APPEND libraries ${ARG_ADDITIONAL_LIBRARIES})

    add_regression_test_executable(NAME ${name}_unaive INPUT ${name} MODE unaive SOURCES ${_sources} LIBRARIES usid_cpu ${libraries})

    if(TARGET GridTools::stencil_gpu)
        add_regression_test_executable(NAME ${name}_ugpu INPUT ${name} MODE ugpu SOURCES ${_sources} LIBRARIES ${libraries} usid_gpu GridTools::stencil_gpu)
        gridtools_setup_target(${name}_ugpu CUDA_ARCH sm_50)
    endif()
endfunction()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    include(FetchGoogleTest.cmake)
    fetch_googletest()

    find_package(Atlas)

    if(Atlas_FOUND)
        add_subdirectory(usid)
        add_subdirectory(gtir)
    endif()
    # add_subdirectory(fortran) # TODO
    
    add_regression_test(cell2cell)
    # add_regression_test(cell2cell_sparse_gtir DRIVER cell2cell) # TODO different driver?!
    add_regression_test(vertex2edge)
    add_regression_test(vertex2edge_gtir DRIVER vertex2edge)
    add_regression_test(sparse_assign_gtir DRIVER sparse_assign)
    add_regression_test(sparse_assign DRIVER sparse_assign) 
    add_regression_test(weights_gtir DRIVER weights)
    add_regression_test(weights DRIVER weights)
    add_regression_test(copy_with_k_gtir DRIVER copy_with_k)
    add_regression_test(copy_with_k DRIVER copy_with_k)
    add_regression_test(tmp_field)
    if(Atlas_FOUND)
        add_regression_test(fvm_nabla ADDITIONAL_LIBRARIES atlas)
    endif()
    add_regression_test(native_functions)
    # add_regression_test(k_fields) #TODO
endif()
