add_library(GridToolsTest INTERFACE)
target_link_libraries(GridToolsTest INTERFACE gridtools)
target_include_directories(GridToolsTest INTERFACE include)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(GridToolsTest INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall
            -Wno-unknown-pragmas
            -Wno-sign-compare
            -Wno-unused-local-typedefs
            -Wno-attributes
            -Wno-unused-but-set-variable
            -Wno-unneeded-internal-declaration
            -Wno-unused-function>)
    target_compile_options(GridToolsTest INTERFACE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler
            -Wall,-Wno-unknown-pragmas,-Wno-sign-compare,-Wno-attributes,-Wno-unused-but-set-variable,-Wno-unneeded-internal-declaration,-Wno-unused-function>")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.4.0")
        target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-value>)
        target_compile_options(GridToolsTest INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-unused-value>")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.9.0")
        # attribute noalias has been added in clang 3.9.0
        target_compile_options(GridToolsTest INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-attributes>)
        target_compile_options(GridToolsTest INTERFACE
                "SHELL:$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wno-unknown-attributes>")
    endif()
endif()

option(GT_TREAT_WARNINGS_AS_ERROR "Treat warnings as errors" OFF)
mark_as_advanced(GT_TREAT_WARNINGS_AS_ERROR)

if(GT_TREAT_WARNINGS_AS_ERROR)
    target_compile_options(GridToolsTest INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror>)
endif()

## cuda support ##
if(TARGET _gridtools_nvcc)
    if(GT_TREAT_WARNINGS_AS_ERROR)
        # unfortunately we cannot treat all as warnings, we have to specify each warning; the only supported warning in CUDA8 is cross-execution-space-call
        # CUDA 9 adds deprecated-declarations (activated) and reorder (not activated)
        target_compile_options(GridToolsTest INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=--warning-as-error>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xnvlink=--warning-as-error>
                $<$<COMPILE_LANGUAGE:CUDA>:-Werror=deprecated-declarations>)
    endif()

    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 9.2)
        # suppress because of warnings in GTest
        target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
    endif()
endif()

function(gridtools_setup_test_target tgt)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES LIBRARIES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    target_sources(${tgt} PRIVATE ${ARGS_SOURCES})
    target_link_libraries(${tgt} PUBLIC ${ARGS_LIBRARIES})
    target_link_libraries(${tgt} PUBLIC GridToolsTest)
    gridtools_setup_target(${tgt} CUDA_ARCH ${GT_CUDA_ARCH})
endfunction()

function(gridtools_add_test_executable tgt)
    add_executable(${tgt})
    gridtools_setup_test_target(${tgt} ${ARGN})
endfunction()


function(gridtools_add_test_library tgt)
    add_library(${tgt})
    gridtools_setup_test_target(${tgt} ${ARGN})
endfunction()

include(FetchGoogletest)
fetch_googletest()
target_link_libraries(GridToolsTest INTERFACE Threads::Threads gtest)

add_subdirectory(src)

function(gridtools_add_mpi_test arch tgt)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES LIBRARIES LABELS)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    gridtools_add_test_executable(${tgt}
            SOURCES ${ARGS_SOURCES}
            LIBRARIES ${ARGS_LIBRARIES} mpi_gtest_main_${arch} gcl_${arch})
    set(nproc 4)
    set(labels ${ARGS_LABELS} mpi gcl ${arch})
    # Note: We use MPITEST_ instead of MPIEXEC_ because our own MPI_TEST_-variables are slurm-aware
    add_test(
            NAME ${tgt}
            COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} ${nproc} ${MPITEST_PREFLAGS} $<TARGET_FILE:${tgt}> ${MPITEST_POSTFLAGS}
    )
    set_tests_properties(${tgt} PROPERTIES LABELS "${labels}")
    set_tests_properties(${tgt} PROPERTIES PROCESSORS ${nproc})
endfunction()


add_subdirectory(regression)
add_subdirectory(unit_tests)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Even if the explicitly requests testing, we cannot run these CMake tests as it would result in an infinite recursion.
    add_subdirectory(cmake)
endif()
