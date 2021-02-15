add_executable(nabla_naive nabla_naive.cpp)
target_link_libraries(nabla_naive PRIVATE usid_cpu gtest_main atlas)
add_test(NAME nabla_naive COMMAND $<TARGET_FILE:nabla_naive>)

if(TARGET usid_gpu)
    add_executable(nabla_gpu nabla_cuda.cu)
    target_link_libraries(nabla_gpu PRIVATE usid_gpu gtest_main atlas)
    add_test(NAME nabla_gpu COMMAND $<TARGET_FILE:nabla_gpu>)
endif()
