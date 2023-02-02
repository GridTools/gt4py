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

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include <cpp_bindgen/array_descriptor.h>

#include "../../common/integral_constant.hpp"
#include "../../common/stride_util.hpp"
#include "../../sid/simple_ptr_holder.hpp"

namespace gridtools {
    namespace fortran_array_view_impl_ {
        struct default_kind {};

        template <class T, size_t Rank, class Kind = default_kind, bool ACCPresent = true>
        class fortran_array_view {
            static_assert(std::is_arithmetic_v<T>, "fortran_array_view should be instantiated with arithmetic type");

            using upper_bounds_t = std::array<ptrdiff_t, Rank>;
            using lower_bounds_t = std::array<integral_constant<ptrdiff_t, 0>, Rank>;
            using strides_t = decltype(stride_util::make_strides_from_sizes(lower_bounds_t()));

            bindgen_fortran_array_descriptor const &m_desc;

            friend sid::simple_ptr_holder<T *> sid_get_origin(fortran_array_view const &obj) {
                return {static_cast<T *>(obj.m_desc.data)};
            }
            friend strides_t sid_get_strides(fortran_array_view const &obj) {
                return stride_util::make_strides_from_sizes(sid_get_upper_bounds(obj));
            }
            friend upper_bounds_t sid_get_upper_bounds(fortran_array_view const &obj) {
                upper_bounds_t res;
                for (size_t i = 0; i != Rank; ++i)
                    res[i] = obj.m_desc.dims[i];
                return res;
            }
            friend Kind sid_get_strides_kind(fortran_array_view const &) { return {}; }
            friend lower_bounds_t sid_get_lower_bounds(fortran_array_view const &) { return {}; }

          public:
            using bindgen_view_rank = std::integral_constant<size_t, Rank>;
            using bindgen_view_element_type = T;
            using bindgen_is_acc_present = std::integral_constant<bool, ACCPresent>;

            fortran_array_view(bindgen_fortran_array_descriptor const &desc) : m_desc(desc) {
#ifndef NDEBUG
                assert(desc.rank == Rank);
                for (size_t i = 0; i != Rank; ++i)
                    assert(desc.dims[i] > 0);
#endif
            }
        };
    } // namespace fortran_array_view_impl_

    // Models both gridtools SID concept and bindgen FortranArrayView concept
    using fortran_array_view_impl_::fortran_array_view;
} // namespace gridtools
