include(FetchContent)
FetchContent_Declare(
        cpp_bindgen
        GIT_REPOSITORY https://github.com/GridTools/cpp_bindgen.git
        GIT_TAG        v1.0.1
)

set(build_testing_ ${BUILD_TESTING})
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(cpp_bindgen)
set(BUILD_TESTING ${build_testing_})

gridtools_add_unit_test(test_fortran_array_adapter
        SOURCES test_fortran_array_adapter.cpp
        LIBRARIES cpp_bindgen_interface
        NO_NVCC)
