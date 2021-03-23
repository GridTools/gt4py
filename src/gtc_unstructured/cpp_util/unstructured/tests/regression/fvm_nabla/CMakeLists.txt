function(add_driver_executable target driver_source stencil_source)
    # generate temporary directory to generate driver code to
    set(gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated_${target})
    file(MAKE_DIRECTORY ${gen_dir})

    # get absolute path to the stencil used in the driver template
    get_filename_component(STENCIL_IMPL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${stencil_source} ABSOLUTE)

    # generate driver source
    get_filename_component(driver_source_file_name "${driver_source}" NAME)
    configure_file(${driver_source} ${gen_dir}/${driver_source_file_name})

    # gridtools_setup_target expects a relative path so transform it
    file(RELATIVE_PATH gen_dir_rel ${CMAKE_CURRENT_SOURCE_DIR} ${gen_dir})

    add_executable(${target} ${gen_dir_rel}/${driver_source_file_name})
endfunction()

add_driver_executable(fvm_nabla driver.cc nabla_naive.hpp)
target_link_libraries(fvm_nabla PRIVATE atlas eckit GridTools::gridtools gtnext gtest gtest_main)
add_test(NAME fvm_nabla COMMAND $<TARGET_FILE:fvm_nabla>)

if(CMAKE_CUDA_COMPILER)
    add_driver_executable(fvm_nabla_cuda driver.cc nabla_cuda.hpp)
    target_link_libraries(fvm_nabla_cuda PRIVATE atlas eckit GridTools::gridtools GridTools::stencil_gpu gtnext gtest gtest_main)
    gridtools_setup_target(fvm_nabla_cuda CUDA_ARCH sm_50)
    target_compile_options(fvm_nabla_cuda PRIVATE "-std=c++17")

    add_test(NAME fvm_nabla_cuda COMMAND $<TARGET_FILE:fvm_nabla_cuda>)
endif()
