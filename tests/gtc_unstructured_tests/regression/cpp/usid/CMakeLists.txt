set(PYTHON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/fvm_nabla_usid.py")
add_custom_command(OUTPUT nabla_naive.hpp nabla_cuda.hpp
    COMMAND ${Python3_EXECUTABLE} ${PYTHON_SOURCE}
    DEPENDS ${PYTHON_SOURCE} 
)

add_executable(nabla_naive nabla_naive.cpp nabla_naive.hpp)
target_link_libraries(nabla_naive PRIVATE usid_cpu gtest_main atlas)
add_test(NAME nabla_naive COMMAND $<TARGET_FILE:nabla_naive>)

if(TARGET usid_gpu)
    add_executable(nabla_gpu nabla_cuda.cu nabla_cuda.hpp)
    target_link_libraries(nabla_gpu PRIVATE usid_gpu gtest_main atlas)
    add_test(NAME nabla_gpu COMMAND $<TARGET_FILE:nabla_gpu>)
endif()
