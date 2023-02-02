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

#include <type_traits>

namespace gridtools {
    namespace meta {
        constexpr size_t find_impl_helper(bool const *first, bool const *last) {
            return first == last || *first ? 0 : 1 + find_impl_helper(first + 1, last);
        }

        template <class, class>
        struct find_impl;

        template <template <class...> class L, class... Ts, class Key>
        struct find_impl<L<Ts...>, Key> {
            static constexpr bool values[sizeof...(Ts)] = {std::is_same_v<Ts, Key>...};
            static constexpr size_t value = find_impl_helper(values, values + sizeof...(Ts));
        };

        template <template <class...> class L, class Key>
        struct find_impl<L<>, Key> {
            static constexpr size_t value = 0;
        };

        template <class List, class Key>
        struct find : std::integral_constant<size_t, find_impl<List, Key>::value> {};
    } // namespace meta
} // namespace gridtools
