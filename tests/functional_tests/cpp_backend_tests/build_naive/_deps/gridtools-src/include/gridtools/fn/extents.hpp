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
 * @file
 *
 * Utility to handle extents.
 *
 * An `extent` is a compile-time type with
 * - a dimension tag
 * - a pair of integral_constants representing relative lower and upper extents
 *
 * An `extents` type is a set of `extent`s.
 * `extents` can be used to extend an int_vector representing offsets (`extend_offsets`) or sizes (`extend_sizes`).
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "../common/int_vector.hpp"
#include "../common/integral_constant.hpp"
#include "../meta.hpp"

namespace gridtools {
    namespace fn {
        template <class Dim, std::ptrdiff_t L, std::ptrdiff_t U>
        struct extent {
            static_assert(L <= U);
            using dim_t = Dim;
            using lower_t = integral_constant<std::ptrdiff_t, L>;
            using upper_t = integral_constant<std::ptrdiff_t, U>;
            using size_t = integral_constant<std::size_t, U - L>;
            using list_t = meta::list<Dim, lower_t, upper_t>;
        };

        template <class>
        struct is_extent : std::false_type {};

        template <class Dim, std::ptrdiff_t L, std::ptrdiff_t U>
        struct is_extent<extent<Dim, L, U>> : std::bool_constant<L <= U> {};

#ifdef __cpp_concepts
        template <class T>
        concept extent_c = is_extent<T>::value;
#endif

        template <class... Ts>
        struct extents {
            static_assert(meta::all_of<is_extent, meta::list<Ts...>>::value);
            static_assert(meta::is_set<meta::list<typename Ts::dim_t...>>::value);
            using keys_t = hymap::keys<typename Ts::dim_t...>;

            static GT_CONSTEVAL GT_FUNCTION auto offsets() {
                return int_vector::prune_zeros(typename keys_t::template values<typename Ts::lower_t...>());
            }
            using offsets_t = decltype(offsets());

            static GT_CONSTEVAL GT_FUNCTION auto sizes() {
                return int_vector::prune_zeros(typename keys_t::template values<typename Ts::size_t...>());
            }
            using sizes_t = decltype(sizes());
        };

        template <class, class = void>
        struct is_extents : std::false_type {};

        template <class... Ts>
        struct is_extents<extents<Ts...>,
            std::enable_if_t<meta::all_of<is_extent, meta::list<Ts...>>::value &&
                             meta::is_set<meta::list<typename Ts::dim_t...>>::value>> : std::true_type {};

#ifdef __cpp_concepts
        template <class T>
        concept extents_c = is_extents<T>::value;
#endif

        template <class Extents, class Offsets>
        decltype(auto) GT_FUNCTION constexpr extend_offsets(Offsets &&src) {
            static_assert(is_extents<Extents>::value);
            static_assert(is_int_vector<std::decay_t<Offsets>>::value);
            using namespace int_vector::arithmetic;
            return std::forward<Offsets>(src) + Extents::offsets();
        }

        template <class Extents, class Sizes>
        decltype(auto) GT_FUNCTION constexpr extend_sizes(Sizes &&sizes) {
            static_assert(is_extents<Extents>::value);
            static_assert(is_int_vector<std::decay_t<Sizes>>::value);
            using namespace int_vector::arithmetic;
            return std::forward<Sizes>(sizes) + Extents::sizes();
        }

        namespace extent_impl_ {
            template <class...>
            struct merge_extents;

            template <class Dim, std::ptrdiff_t... L, std::ptrdiff_t... U>
            struct merge_extents<meta::list<Dim, extent<Dim, L, U>>...> {
                using type = meta::list<Dim, extent<Dim, std::min({L...}), std::max({U...})>>;
            };
        } // namespace extent_impl_

        // T any number of individual `extent`s and produce the normalized `extents`.
        // If some `extent`s have the same dimension, they are merged.
        template <class... Extents>
        using make_extents = meta::rename<extents,
            meta::transform<meta::second,
                meta::mp_make<meta::force<extent_impl_::merge_extents>::template apply,
                    meta::list<meta::list<typename Extents::dim_t, Extents>...>>>>;

        // Merge several `extents`s into one
        template <class... Extentss>
        using enclosing_extents = meta::rename<make_extents, meta::concat<meta::rename<meta::list, Extentss>...>>;

    } // namespace fn
} // namespace gridtools
