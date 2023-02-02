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

namespace gridtools {
    using int_t = int;
    using uint_t = unsigned int;
} // namespace gridtools

#if defined(__CUDACC__)
#define GT_CUDACC
#ifdef __CUDA_ARCH__
#define GT_CUDA_ARCH __CUDA_ARCH__
#endif
#elif defined(__HIP__)
#define GT_CUDACC
#ifdef __HIP_DEVICE_COMPILE__
#define GT_CUDA_ARCH 1
#endif
#endif

#ifdef __cpp_consteval
#define GT_CONSTEVAL consteval
#else
#define GT_CONSTEVAL constexpr
#endif

#define GT_INTERNAL_ERROR                                                                                       \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team."

#define GT_INTERNAL_ERROR_MSG(x) GT_INTERNAL_ERROR "\nMessage\n\n" x

#ifdef __NVCC__
#define GT_NVCC_DIAG_STR(x) #x
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#define GT_NVCC_DIAG_PUSH_SUPPRESS(x) _Pragma("nv_diagnostic push") _Pragma(GT_NVCC_DIAG_STR(nv_diag_suppress x))
#define GT_NVCC_DIAG_POP_SUPPRESS(x) _Pragma("nv_diagnostic pop")
#else
#define GT_NVCC_DIAG_PUSH_SUPPRESS(x) _Pragma(GT_NVCC_DIAG_STR(diag_suppress = x))
#define GT_NVCC_DIAG_POP_SUPPRESS(x) _Pragma(GT_NVCC_DIAG_STR(diag_default = x))
#endif
#else
#define GT_NVCC_DIAG_PUSH_SUPPRESS(x)
#define GT_NVCC_DIAG_POP_SUPPRESS(x)
#endif
