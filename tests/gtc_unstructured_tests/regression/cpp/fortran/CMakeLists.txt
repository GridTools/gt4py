enable_language(Fortran)

include(FetchContent)
FetchContent_Declare(
    cpp_bindgen
    GIT_REPOSITORY https://github.com/GridTools/cpp_bindgen.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(cpp_bindgen)

find_package(OpenACC REQUIRED)

bindgen_add_library(vertex2edge_lib SOURCES vertex2edge_wrapper.cc)
target_link_libraries(vertex2edge_lib PUBLIC GridTools::gridtools usid_cpu cpp_bindgen_interface)

add_executable(vertex2edge_ftn_driver driver.f90)
target_link_libraries(vertex2edge_ftn_driver vertex2edge_lib_fortran)
set_target_properties(vertex2edge_ftn_driver PROPERTIES LINKER_LANGUAGE Fortran)


if(OpenACC_Fortran_FOUND AND TARGET GridTools::stencil_gpu)
    bindgen_add_library(vertex2edge_cuda_lib SOURCES vertex2edge_wrapper.cu FORTRAN_MODULE_NAME vertex2edge_lib)
    target_link_libraries(vertex2edge_cuda_lib PUBLIC GridTools::gridtools GridTools::stencil_gpu usid_gpu cpp_bindgen_interface)
    target_link_libraries(vertex2edge_cuda_lib_fortran PUBLIC OpenACC::OpenACC_Fortran)

    add_executable(vertex2edge_ftn_cuda_driver driver.f90)
    target_link_libraries(vertex2edge_ftn_cuda_driver vertex2edge_cuda_lib_fortran OpenACC::OpenACC_Fortran)
    set_target_properties(vertex2edge_ftn_cuda_driver PROPERTIES LINKER_LANGUAGE Fortran)
endif()
