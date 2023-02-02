# try_clang_cuda()
# Parameters:
#    - gt_result: result variable is set to Clang-CUDA or NOTFOUND
function(try_clang_cuda gt_result)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        include(try_compile_clang_cuda)
        try_compile_clang_cuda(GT_CLANG_CUDA_WORKS "sm_60")
        if(GT_CLANG_CUDA_WORKS)
            set(${gt_result} Clang-CUDA PARENT_SCOPE)
            return()
        endif()
    endif()
    set(${gt_result} NOTFOUND PARENT_SCOPE)
endfunction()

# try_nvcc_cuda()
# Parameters:
#    - gt_result: result variable is set to NVCC-CUDA or NOTFOUND
function(try_nvcc_cuda gt_result)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(${gt_result} NVCC-CUDA PARENT_SCOPE)
        return()
    endif()
    set(${gt_result} NOTFOUND PARENT_SCOPE)
endfunction()

# detect_cuda_type()
# Parameters:
#    - cuda_type: result variable is set to one of HIPCC-AMDGPU/NVCC-CUDA/Clang-CUDA/NOTFOUND
#    - clang_mode: AUTO, Clang-CUDA, NVCC-CUDA
#       - AUTO: Prefer NVCC-CUDA if the CUDA language is enabled, else try Clang-CUDA
#       - Clang-CUDA: Try Clang-CUDA or fail.
#       - NVCC-CUDA: Try NVCC-CUDA or fail.
function(detect_cuda_type cuda_type clang_mode)
    get_filename_component(cxx_name ${CMAKE_CXX_COMPILER} NAME)
    if(cxx_name STREQUAL "hipcc")
        include(try_compile_hip)
        try_compile_hip(GT_HIP_WORKS) #TODO use cache variable to avoid compiling each cmake run
        if(GT_HIP_WORKS)
            set(${cuda_type} HIPCC-AMDGPU PARENT_SCOPE)
            return()
        else()
            message(FATAL_ERROR "${cxx_name} wasn't able to compile a simple HIP program.")
        endif()
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        try_nvcc_cuda(gt_result)
        set(${cuda_type} ${gt_result} PARENT_SCOPE)
        return()
    else() # Clang
        string(TOLOWER "${clang_mode}" _lower_case_clang_cuda)
        if(_lower_case_clang_cuda STREQUAL "clang-cuda")
            try_clang_cuda(gt_result)
            if(gt_result)
                set(${cuda_type} ${gt_result} PARENT_SCOPE)
                return()
            else()
                message(FATAL_ERROR "Clang-CUDA mode was selected, but doesn't work.")
            endif()
        elseif(_lower_case_clang_cuda STREQUAL "nvcc-cuda")
            try_nvcc_cuda(gt_result)
            if(gt_result)
                set(${cuda_type} ${gt_result} PARENT_SCOPE)
                return()
            else()
                message(FATAL_ERROR "NVCC-CUDA mode was selected, but doesn't work.")
            endif()
        elseif(_lower_case_clang_cuda STREQUAL "auto") # AUTO
            get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
            if("CUDA" IN_LIST languages) # CUDA language is already enabled, prefer it
                set(${cuda_type} NVCC-CUDA PARENT_SCOPE)
                return()
            else()
                # Prefer Clang-CUDA
                try_clang_cuda(gt_result)
                if(gt_result)
                    set(${cuda_type} ${gt_result} PARENT_SCOPE)
                    return()
                endif()

                # Clang-CUDA doesn't work, try NVCC
                try_nvcc_cuda(gt_result)
                if(gt_result)
                    set(${cuda_type} ${gt_result} PARENT_SCOPE)
                    return()
                endif()

                set(${cuda_type} NOTFOUND PARENT_SCOPE)
            endif()
        else()
            message(FATAL_ERROR "Clang CUDA mode set to invalid value ${clang_mode}")
        endif()
    endif()
    set(${cuda_type} NOTFOUND PARENT_SCOPE)
endfunction()
