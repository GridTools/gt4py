add_executable(nabla_naive_gtir nabla_naive.cpp)
target_link_libraries(nabla_naive_gtir PRIVATE usid_cpu gtest_main atlas)
add_test(NAME nabla_naive_gtir COMMAND $<TARGET_FILE:nabla_naive_gtir>)

if(TARGET usid_gpu)
    add_executable(nabla_gpu_gtir nabla_cuda.cu)
    target_link_libraries(nabla_gpu_gtir PRIVATE usid_gpu gtest_main atlas)
    add_test(NAME nabla_gpu_gtir COMMAND $<TARGET_FILE:nabla_gpu_gtir>)
endif()
