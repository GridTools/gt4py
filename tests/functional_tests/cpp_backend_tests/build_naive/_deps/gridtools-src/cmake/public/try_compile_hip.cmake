# try_compile_hip(result)
# Parameters:
#    - result: result variable is set to TRUE if HIP compilation worked.
function(try_compile_hip result)
    set(HIP_TEST_SOURCE
"
#include <hip/hip_runtime.h>
__global__ void helloworld(int* in, int* out) {
    *out = *in;
}
int main(int argc, char* argv[]) {
    int* in;
    int* out;
    hipMalloc((void**)&in, sizeof(int));
    hipMalloc((void**)&out, sizeof(int));
    helloworld<<<1,1>>>(in, out);
    hipFree(in);
    hipFree(out);
}
")

    set(SRC_FILE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/TryHip.cpp)
    file(WRITE "${SRC_FILE}" "${HIP_TEST_SOURCE}")

    try_compile(hip_works ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY} ${SRC_FILE}
        OUTPUT_VARIABLE HIP_TRY_COMPILE_OUTPUT
        )
    if(NOT hip_works)
        file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
            "Testing HIP compilation gave the following output:\n${HIP_TRY_COMPILE_OUTPUT}\n\n")
    endif()
    set(${result} ${hip_works} PARENT_SCOPE)
endfunction()
