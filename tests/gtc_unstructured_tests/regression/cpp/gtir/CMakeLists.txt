set(PYTHON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/fvm_nabla_gtir.py")
add_custom_command(OUTPUT nabla_naive.hpp nabla_cuda.hpp
    COMMAND ${Python3_EXECUTABLE} ${PYTHON_SOURCE}
    DEPENDS ${PYTHON_SOURCE} 
)

add_executable(nabla_naive_gtir nabla_naive.cpp nabla_naive.hpp)
target_link_libraries(nabla_naive_gtir PRIVATE usid_cpu gtest_main atlas)
add_test(NAME nabla_naive_gtir COMMAND $<TARGET_FILE:nabla_naive_gtir>)

if(TARGET usid_gpu)
    add_executable(nabla_gpu_gtir nabla_cuda.cu nabla_cuda.hpp)
    target_link_libraries(nabla_gpu_gtir PRIVATE usid_gpu gtest_main atlas)
    add_test(NAME nabla_gpu_gtir COMMAND $<TARGET_FILE:nabla_gpu_gtir>)
endif()
