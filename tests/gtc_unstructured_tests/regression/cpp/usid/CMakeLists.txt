set(PYTHON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/fvm_nabla_usid.py")

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/nabla_unaive.hpp
    COMMAND ${Python3_EXECUTABLE} ${PYTHON_SOURCE}
    DEPENDS ${PYTHON_SOURCE}
)
        
add_executable(nabla_naive_usid nabla_naive.cpp ${CMAKE_CURRENT_SOURCE_DIR}/nabla_unaive.hpp)
target_link_libraries(nabla_naive_usid PRIVATE usid_cpu gtest_main atlas)
add_test(NAME nabla_naive_usid COMMAND $<TARGET_FILE:nabla_naive_usid>)
        
if(TARGET usid_gpu)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/nabla_ugpu.hpp
            COMMAND ${Python3_EXECUTABLE} ${PYTHON_SOURCE} ugpu
            DEPENDS ${PYTHON_SOURCE}
    )
    add_executable(nabla_gpu_usid nabla_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/nabla_ugpu.hpp)
    target_link_libraries(nabla_gpu_usid PRIVATE usid_gpu gtest_main atlas)
    add_test(NAME nabla_gpu_usid COMMAND $<TARGET_FILE:nabla_gpu_usid>)
endif()
