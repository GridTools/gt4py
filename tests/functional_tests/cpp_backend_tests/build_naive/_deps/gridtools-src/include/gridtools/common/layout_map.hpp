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

#include <algorithm>
#include <type_traits>

#include "../meta/combine.hpp"
#include "../meta/filter.hpp"
#include "../meta/length.hpp"
#include "../meta/list.hpp"
#include "../meta/macros.hpp"
#include "../meta/push_back.hpp"
#include "defs.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \defgroup layout Layout Map
        @{
    */

    namespace layout_map_impl {
        template <class I>
        using not_negative = std::bool_constant<(I::value >= 0)>;
        template <class A, class B>
        using integral_plus = std::integral_constant<int, A::value + B::value>;

        template <int... Args>
        class layout_map {
            /* list of all arguments */
            using args = meta::list<std::integral_constant<int, Args>...>;

            /* list of all unmasked (i.e. non-negative) arguments */
            using unmasked_args = meta::filter<layout_map_impl::not_negative, args>;

            /* sum of all unmasked arguments (only used for assertion below) */
            static constexpr int unmasked_arg_sum = meta::lazy::combine<layout_map_impl::integral_plus,
                meta::push_back<unmasked_args, std::integral_constant<int, 0>>>::type::value;

          public:
            static constexpr int max_arg = std::max({Args...});

            /** @brief Length of layout map excluding masked dimensions. */
            static constexpr std::size_t unmasked_length = meta::length<unmasked_args>::value;
            /** @brief Total length of layout map, including masked dimensions. */
            static constexpr std::size_t masked_length = sizeof...(Args);

            static_assert(unmasked_arg_sum == unmasked_length * (unmasked_length - 1) / 2,
                GT_INTERNAL_ERROR_MSG("Layout map args must not contain any holes (e.g., layout_map<3,1,0>)."));

            /** @brief Get the position of the element with value `i` in the layout map. */
            static constexpr std::size_t find(int i) {
                int args[] = {Args...};
                std::size_t res = 0;
                for (; res != sizeof...(Args); ++res)
                    if (i == args[res])
                        break;
                return res;
            }

            /** @brief Get the value of the element at position `I` in the layout map. */
            static constexpr int at(std::size_t i) {
                int args[] = {Args...};
                return args[i];
            }
        };

        template <class>
        struct reverse_map;

        template <int... Is>
        struct reverse_map<layout_map<Is...>> {
            static constexpr int max = std::max({Is...});
            using type = layout_map<(Is < 0 ? Is : max - Is)...>;
        };

        template <class, class>
        struct layout_transform;

        template <class Layout, int... Is>
        struct layout_transform<Layout, layout_map<Is...>> {
            using type = layout_map<Layout::at(Is)...>;
        };

    } // namespace layout_map_impl
    using layout_map_impl::layout_map;
    template <class T>
    using reverse_map = typename layout_map_impl::reverse_map<T>::type;
    template <class T, class U>
    using layout_transform = typename layout_map_impl::layout_transform<T, U>::type;
} // namespace gridtools
