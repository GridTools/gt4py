/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_GENERIC_METAFUNCTIONS_FOR_EACH_HPP_
#define GT_COMMON_GENERIC_METAFUNCTIONS_FOR_EACH_HPP_

#include "host_device.hpp"

#define GT_FILENAME <gridtools/common/for_each.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {
    GT_TARGET_NAMESPACE {
        namespace for_each_impl_ {
            template <class>
            struct for_each_f;

            template <template <class...> class L, class... Ts>
            struct for_each_f<L<Ts...>> {
                template <class F>
                GT_TARGET GT_FORCE_INLINE constexpr void operator()(F &&f) const {
                    (..., f(Ts()));
                }
            };
        } // namespace for_each_impl_

        template <class L>
#if defined(GT_TARGET_HAS_DEVICE) and defined(__NVCC__)
        GT_DEVICE
#endif
        constexpr for_each_impl_::for_each_f<L> for_each = {};
    }
} // namespace gridtools

#endif
