/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
@file
@brief Unstructured collection of small generic purpose functors and related helpers.

   All functors here are supplied with the inner result or result_type to follow boost::result_of requirement
   for the cases if BOOST_RESULT_OF_USE_DECLTYPE is not defined [It is so for nvcc8]. This makes those functors
   usable in the context of boost::fuison high order functions.

*/

#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_FUNCTIONAL_HPP_
#define GT_COMMON_FUNCTIONAL_HPP_

#include <utility>

#include "host_device.hpp"

#define GT_FILENAME <gridtools/common/functional.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup functional Functional
        @{
    */
    GT_TARGET_NAMESPACE {

        /// Forward the args to constructor.
        //
        template <typename T>
        struct ctor {
            template <typename... Args>
            GT_TARGET GT_FORCE_INLINE constexpr T operator()(Args &&...args) const {
                return {std::forward<Args>(args)...};
            }
        };

        /// Do nothing.
        //
        struct noop {
            template <typename... Args>
            GT_TARGET GT_FORCE_INLINE constexpr void operator()(Args &&...) const {}
        };

        /// Perfectly forward the argument.
        //
        struct identity {
            template <typename Arg>
            GT_TARGET GT_FORCE_INLINE constexpr Arg operator()(Arg &&arg) const {
                return arg;
            }
        };

        /// Copy the argument.
        //
        struct clone {
            template <typename Arg>
            GT_TARGET GT_FORCE_INLINE constexpr Arg operator()(Arg const &arg) const {
                return arg;
            }
        };

        template <class... Fs>
        struct overload : Fs... {
            constexpr GT_TARGET GT_FORCE_INLINE overload(Fs... fs) : Fs(std::move(fs))... {}
            using Fs::operator()...;
        };
    }
    /** @} */
    /** @} */
} // namespace gridtools

#endif
