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

#include <cmath>

#include "defs.hpp"
#include "host_device.hpp"

#ifdef GT_CUDACC
#include "atomic_cuda.hpp"
#endif
#include "atomic_host.hpp"

/**
 * Namespace providing a set of atomic functions working for all backends
 */
namespace gridtools {
    /** \ingroup common
        @{
     */

    /** \defgroup atomic Atomic Functions
        @{
    */

#ifdef __NVCC__
#ifdef GT_CUDA_ARCH
#define GT_DECLARE_ATOMIC(name)                               \
    template <class T>                                        \
    GT_FUNCTION_DEVICE T atomic_##name(T &var, const T val) { \
        return atomic_cuda<T>::atomic_##name(var, val);       \
    }
#else
#define GT_DECLARE_ATOMIC(name)                             \
    template <class T>                                      \
    GT_FUNCTION_HOST T atomic_##name(T &var, const T val) { \
        return atomic_host<T>::atomic_##name(var, val);     \
    }
#endif
#else
#ifdef GT_CUDACC
#define GT_DECLARE_ATOMIC(name)                                       \
    template <class T>                                                \
    GT_FORCE_INLINE __device__ T atomic_##name(T &var, const T val) { \
        return atomic_cuda<T>::atomic_##name(var, val);               \
    }                                                                 \
    template <class T>                                                \
    GT_FUNCTION_HOST T atomic_##name(T &var, const T val) {           \
        return atomic_host<T>::atomic_##name(var, val);               \
    }
#else
#define GT_DECLARE_ATOMIC(name)                             \
    template <class T>                                      \
    GT_FUNCTION_HOST T atomic_##name(T &var, const T val) { \
        return atomic_host<T>::atomic_##name(var, val);     \
    }
#endif
#endif

    GT_DECLARE_ATOMIC(add)
    GT_DECLARE_ATOMIC(sub)
    GT_DECLARE_ATOMIC(exch)
    GT_DECLARE_ATOMIC(min)
    GT_DECLARE_ATOMIC(max)

#undef GT_DECLARE_ATOMIC

    /** @} */
    /** @} */
} // namespace gridtools
