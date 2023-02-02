string(TOLOWER "${CMAKE_BUILD_TYPE}" PYUTILS_BUILD_TYPE)

if(TARGET _gridtools_cuda)
    if (TARGET _gridtools_nvcc)
        set(PYUTILS_COMPILER "${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION})")
    else()
        set(PYUTILS_COMPILER "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION})")
    endif()
else()
    set(PYUTILS_COMPILER "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

if(TARGET stencil_cpu_kfirst)
    list(APPEND PYUTILS_BACKEND "cpu_kfirst")
endif()

if(TARGET stencil_naive)
    list(APPEND PYUTILS_BACKEND "naive")
endif()

if(TARGET stencil_gpu)
    list(APPEND PYUTILS_BACKEND "gpu")
endif()

if(TARGET stencil_cpu_ifirst)
    list(APPEND PYUTILS_BACKEND "cpu_ifirst")
endif()

file(COPY . DESTINATION . FILES_MATCHING PATTERN *.py)
configure_file(pyutils/buildinfo.py.in pyutils/buildinfo.py)
