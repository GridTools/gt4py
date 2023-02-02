find_package(Python3 COMPONENTS Interpreter Development NumPy)

if (NOT Python3_FOUND OR NOT Python3_Development_FOUND OR NOT Python3_NumPy_FOUND)
    return()
endif()

# This option is mostly useful if we have a broken setup
option(GT_TESTS_ENABLE_PYTHON_TESTS "Enable Python tests" ON)

if(NOT GT_TESTS_ENABLE_PYTHON_TESTS)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.10.0
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

pybind11_add_module(py_implementation implementation.cpp)

target_link_libraries(py_implementation PRIVATE gridtools)

add_test(NAME py_bindings COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driver.py)
