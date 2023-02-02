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
#include "integral_constant.hpp"
#include "tuple_util.hpp"

namespace gridtools {
    namespace stride_util {
        namespace _impl {

            template <size_t I, class Sizes>
            struct stride_type {
                using type = decltype(tuple_util::element<I - 1, Sizes>() * typename stride_type<I - 1, Sizes>::type());
            };

            template <class Sizes>
            struct stride_type<0, Sizes> : integral_constant<int_t, 1> {};

            template <size_t I, class Sizes, std::enable_if_t<I == 0, int> = 0>
            integral_constant<int_t, 1> get_stride(Sizes const &) {
                return {};
            }

            template <size_t I, class Sizes, std::enable_if_t<I != 0, int> = 0>
            typename stride_type<I, Sizes>::type get_stride(Sizes const &sizes) {
                return tuple_util::get<I - 1>(sizes) * get_stride<I - 1>(sizes);
            }

            template <class Sizes>
            struct from_size_to_stride_f {
                Sizes const &m_sizes;

                template <size_t I, class Size>
                auto operator()(Size &&) const {
                    return get_stride<I>(m_sizes);
                }
            };
        } // namespace _impl

        template <class Sizes>
        auto make_strides_from_sizes(Sizes const &sizes) {
            return tuple_util::transform_index(_impl::from_size_to_stride_f<Sizes>{sizes}, sizes);
        }

        template <class Sizes>
        auto total_size(Sizes const &sizes) {
            return tuple_util::fold([](auto l, auto r) { return l * r; }, sizes);
        }
    } // namespace stride_util
} // namespace gridtools
