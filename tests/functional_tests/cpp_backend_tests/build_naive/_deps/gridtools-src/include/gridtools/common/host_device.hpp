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
/**
@file
@brief definition of macros for host/GPU
*/
/** \ingroup common
    @{
    \defgroup hostdevice Host-Device Macros
    @{
*/

#include "defs.hpp"

#ifdef __HIPCC__
#include "cuda_runtime.hpp"
#endif

#if defined(__NVCC__)
#define GT_FORCE_INLINE __forceinline__
#define GT_FORCE_INLINE_LAMBDA
#elif defined(__GNUC__)
#define GT_FORCE_INLINE inline __attribute__((always_inline))
#define GT_FORCE_INLINE_LAMBDA __attribute__((always_inline))
#elif defined(_MSC_VER)
#define GT_FORCE_INLINE inline __forceinline
#define GT_FORCE_INLINE_LAMBDA
#else
#define GT_FORCE_INLINE inline
#define GT_FORCE_INLINE_LAMBDA
#endif

/**
 * @def GT_FUNCTION
 * Function attribute macro to be used for host-device functions.
 */
/**
 * @def GT_FUNCTION_HOST
 * Function attribute macro to be used for host-only functions.
 */
/**
 * @def GT_FUNCTION_DEVICE
 * Function attribute macro to be used for device-only functions.
 */
/**
 * @def GT_FUNCTION_WARNING
 * Function attribute macro to be used for host-only functions that might call a host-device
 * function. This macro is only needed to supress NVCC warnings.
 */

#ifdef GT_CUDACC
#define GT_HOST_DEVICE __host__ __device__
#ifdef __NVCC__ // NVIDIA CUDA compilation
#define GT_DEVICE __device__
#define GT_HOST __host__
#else // Clang CUDA or HIP compilation
#define GT_DEVICE __device__ __host__
#define GT_HOST __host__
#endif
#else
#define GT_HOST_DEVICE
#define GT_HOST
#endif

#ifndef GT_FUNCTION
#define GT_FUNCTION GT_HOST_DEVICE GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_WARNING
#define GT_FUNCTION_WARNING GT_HOST_DEVICE GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_HOST
#define GT_FUNCTION_HOST GT_HOST GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_DEVICE
#define GT_FUNCTION_DEVICE GT_DEVICE GT_FORCE_INLINE
#endif

/**
 *   A helper to implement a family of functions which are different from each other only by target specifies.
 *
 *   It uses the same design pattern as `BOOST_PP_ITERATE` does.
 *   For example if one wants to define a function with any possible combination of `__host__` and `__device__`
 *   specifiers he needs to write the following code:
 *
 *   foo.hpp:
 *
 *   \code
 *   // here we query if this file is used in the context of iteration
 *   #ifndef GT_TARGET_ITERATING
 *
 *   // note that you can't use `#pragma once` here and have to use classic header guards instead
 *   #ifndef FOO_HPP_
 *   #define FOO_HPP_
 *
 *   #include <path/to/this/file/host_device.hpp>
 *
 *   // we need to provide GT_ITERATE_ON_TARGETS() with the name of the current file to include it back during
 *   // iteration process. GT_FILENAME is a hardcoded name that GT_ITERATE_ON_TARGETS() will use.
 *   #define GT_FILENAME <path/to/the/user/file/foo.hpp>
 *
 *   // iteration takes place here
 *   #include GT_ITERATE_ON_TARGETS()
 *
 *   // cleanup
 *   #undef GT_FILENAME
 *
 *   #endif
 *   #else
 *
 *   // here is the code that will be included several times during the iteration process
 *
 *   namespace my {
 *     // GT_TARGET_NAMESPACE will be defined by GT_ITERATE_ON_TARGETS() for you.
 *     // It could be either `namespace host` or `namespace device` or `namespace host_device`.
 *     // one of those namespaces would be defined as `inline`
 *     GT_TARGET_NAMESPACE {
 *        // GT_TARGET will be defined by GT_ITERATE_ON_TARGETS() for you.
 *        // It will contain target specifier that is needed it the given context.
 *        GT_TARGET void foo() {}
 *     }
 *   }
 *
 *   #endif
 *   \endcode
 *
 *   By including "file.hpp" file the following symbols would be available:
 *   - `my::foo`
 *   - `my::host::foo`
 *   - `my::device::foo`
 *   - `my::host_device::foo`
 *
 *   where:
 *
 *   - `my::host::foo` has no specifiers.
 *   - `my::foo` is resolved to `my::host::foo`
 *
 *   If compiling with CUDA, `my::device::foo` has `__device__` specifier, `my::host_device::foo` has
 *   `__host__ __device__` specifier.
 *
 *   Otherwise `my::device::foo` and `my::host_device::foo` are resolved to `my::host::foo`.
 */
#define GT_ITERATE_ON_TARGETS() <gridtools/common/iterate_on_host_device.hpp>

/** @} */
/** @} */
