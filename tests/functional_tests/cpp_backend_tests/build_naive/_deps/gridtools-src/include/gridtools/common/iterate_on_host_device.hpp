/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// DON'T USE #pragma once HERE!!!

#if !defined(GT_FILENAME)
#error GT_FILENAME is not defined
#endif

#if defined(GT_TARGET_ITERATING)
#error nesting target iterating is not supported
#endif

#if defined(GT_TARGET)
#error GT_TARGET should not be defined outside of this file
#endif

#if defined(GT_TARGET_NAMESPACE)
#error GT_TARGET_NAMESPACE should not be defined outside of this file
#endif

#define GT_TARGET_ITERATING

#ifdef GT_CUDACC

#define GT_TARGET_NAMESPACE_NAME host
#define GT_TARGET_NAMESPACE inline namespace host
#define GT_TARGET GT_HOST

#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME

#define GT_TARGET_NAMESPACE_NAME host_device
#define GT_TARGET_NAMESPACE namespace host_device
#define GT_TARGET GT_HOST_DEVICE
#define GT_TARGET_HAS_DEVICE
#include GT_FILENAME
#undef GT_TARGET_HAS_DEVICE
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME

#define GT_TARGET_NAMESPACE_NAME device
#define GT_TARGET_NAMESPACE namespace device
#define GT_TARGET GT_DEVICE
#define GT_TARGET_HAS_DEVICE
#include GT_FILENAME
#undef GT_TARGET_HAS_DEVICE
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME

#else

#define GT_TARGET_NAMESPACE_NAME host
#define GT_TARGET_NAMESPACE   \
    inline namespace host {}  \
    namespace host_device {   \
        using namespace host; \
    }                         \
    inline namespace host
#define GT_TARGET GT_HOST
#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME
#endif

#undef GT_TARGET_ITERATING
