/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "defs.hpp"

#ifdef GT_CUDACC

#include "cuda_runtime.hpp"
#include "cuda_util.hpp"

namespace gridtools {
    /**
     * @brief returns true if ptr is pointing to CUDA device memory
     * @warning A ptr which is not allocated by cudaMalloc, cudaMallocHost, ... (a normal cpu ptr) will emit an error in
     * cuda-memcheck.
     */
    inline bool is_gpu_ptr(void const *ptr) {
        cudaPointerAttributes ptrAttributes;
        cudaError_t error = cudaPointerGetAttributes(&ptrAttributes, ptr);
        if (error == cudaSuccess)

#if defined(CUDART_VERSION) && CUDART_VERSION < 10000 or defined(__HIPCC__)
            return ptrAttributes.memoryType == cudaMemoryTypeDevice; // deprecated in CUDA 10
#else
            return ptrAttributes.type == cudaMemoryTypeDevice || ptrAttributes.type == cudaMemoryTypeManaged;
#endif
        if (error != cudaErrorInvalidValue)
            GT_CUDA_CHECK(error);

        cudaGetLastError(); // clear the error code
        return false;       // it is not a ptr allocated with cudaMalloc, cudaMallocHost, ...
    }
} // namespace gridtools

#else

namespace gridtools {
    inline bool is_gpu_ptr(void const *) { return false; }
} // namespace gridtools

#endif
