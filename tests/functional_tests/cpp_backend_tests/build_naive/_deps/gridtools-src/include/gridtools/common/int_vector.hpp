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
 * Implements operations on `int_vector`s, which are hymaps of integers (integral types or integral_constants)
 *
 * A type `T` is `int_vector` if
 *  - `is_hymap<T>`
 *  - all value types `V` are std::is_integral<V> or is_gr_integral_constant<V>
 *  - `std::is_trivially_copy_constructible<T>`
 */

#pragma once

#include <type_traits>
#include <utility>

#include "../meta.hpp"
#include "defs.hpp"
#include "host_device.hpp"
#include "hymap.hpp"
#include "integral_constant.hpp"
#include "tuple_util.hpp"

namespace gridtools {

    namespace int_vector::impl_ {
        template <class... Ts>
        struct value_t_merger;

        template <template <class...> class Item, class Key, class... Vs>
        struct value_t_merger<Item<Key, Vs>...> {
            using type = Item<Key, std::decay_t<decltype((Vs{} + ...))>>;
        };

        template <class... Ts>
        using merger_t = typename value_t_merger<Ts...>::type;

        template <class Key>
        struct add_f {
            template <class... Ts>
            GT_FUNCTION constexpr decltype(auto) operator()(Ts const &...args) const {
                return (host_device::at_key_with_default<Key, integral_constant<int, 0>>(args) + ...);
            }
        };

        template <class I>
        using is_constant_zero = is_integral_constant_of<meta::second<I>, 0>;

        template <class Key>
        struct at_key_f {
            template <class T>
            GT_FUNCTION constexpr decltype(auto) operator()(T const &arg) const {
                return host_device::at_key<Key>(arg);
            }
        };

        template <class T>
        using is_integral_or_gr_integral_constant =
            std::bool_constant<std::is_integral_v<T> || is_gr_integral_constant<T>::value>;

        template <class T>
        using elements_are_integral_or_gr_integral_constant =
            meta::all_of<is_integral_or_gr_integral_constant, tuple_util::traits::to_types<T>>;

        template <class T, class = void>
        struct is_int_vector : std::false_type {};

        template <class T>
        struct is_int_vector<T,
            std::enable_if_t<is_hymap<T>::value &&
                             int_vector::impl_::elements_are_integral_or_gr_integral_constant<T>::value>>
            : std::true_type {};
    } // namespace int_vector::impl_

    using int_vector::impl_::is_int_vector;

    template <class T>
    constexpr bool is_int_vector_v = is_int_vector<T>::value;

#ifdef __cpp_concepts
    template <class T>
    concept int_vector_c = is_int_vector<T>::value;
#endif

    namespace int_vector {
        /**
         * @brief Returns elementwise sum of `int_vector`s
         *
         * The keys of the resulting `int_vector` are the union of the keys of the operands.
         */
        template <class... Vecs>
        GT_FUNCTION constexpr auto plus(Vecs && ...vecs) {
            using merged_meta_map_t = meta::mp_make<impl_::merger_t, meta::concat<hymap::to_meta_map<Vecs>...>>;
            using keys_t = meta::transform<meta::first, merged_meta_map_t>;
            using generators = meta::transform<impl_::add_f, keys_t>;
            return tuple_util::host_device::generate<generators, hymap::from_meta_map<merged_meta_map_t>>(
                std::forward<Vecs>(vecs)...);
        }

        /**
         * @brief Returns `int_vector` with elements multiplied by an integral scalar
         */
        template <class Vec, class Scalar>
        GT_FUNCTION constexpr auto multiply(Vec && vec, Scalar scalar) {
            return tuple_util::host_device::transform([scalar](auto v) { return v * scalar; }, std::forward<Vec>(vec));
        }

        /**
         * @brief Returns `int_vector` with elements removed that are `integral_constant<T, 0>`
         */
        template <class Vec>
        GT_FUNCTION constexpr auto prune_zeros(Vec && vec) {
            using filtered_map_t = meta::filter<meta::not_<impl_::is_constant_zero>::apply, hymap::to_meta_map<Vec>>;
            using keys_t = meta::transform<meta::first, filtered_map_t>;
            using generators = meta::transform<impl_::at_key_f, keys_t>;
            return tuple_util::host_device::generate<generators, hymap::from_meta_map<filtered_map_t>>(
                std::forward<Vec>(vec));
        }

        namespace arithmetic {
            template <class Vec,
                class Scalar,
                std::enable_if_t<is_int_vector_v<std::decay_t<Vec>> &&
                                     impl_::is_integral_or_gr_integral_constant<Scalar>::value,
                    bool> = true>
            GT_FUNCTION constexpr auto operator*(Vec &&vec, Scalar scalar) {
                return multiply(std::forward<Vec>(vec), scalar);
            }

            template <class Vec,
                class Scalar,
                std::enable_if_t<is_int_vector_v<std::decay_t<Vec>> &&
                                     impl_::is_integral_or_gr_integral_constant<Scalar>::value,
                    bool> = true>
            GT_FUNCTION constexpr auto operator*(Scalar scalar, Vec &&vec) {
                return multiply(std::forward<Vec>(vec), scalar);
            }

            template <class First,
                class Second,
                std::enable_if_t<is_int_vector_v<std::decay_t<First>> && is_int_vector_v<std::decay_t<Second>>, bool> =
                    true>
            GT_FUNCTION constexpr auto operator+(First &&first, Second &&second) {
                return plus(std::forward<First>(first), std::forward<Second>(second));
            }

            template <class Vec, std::enable_if_t<is_int_vector_v<std::decay_t<Vec>>, bool> = true>
            GT_FUNCTION constexpr auto operator+(Vec &&vec) {
                return vec;
            }

            template <class Vec, std::enable_if_t<is_int_vector_v<std::decay_t<Vec>>, bool> = true>
            GT_FUNCTION constexpr auto operator-(Vec &&vec) {
                return multiply(std::forward<Vec>(vec), integral_constant<int, -1>{});
            }

            template <class First,
                class Second,
                std::enable_if_t<is_int_vector_v<std::decay_t<First>> && is_int_vector_v<std::decay_t<Second>>, bool> =
                    true>
            GT_FUNCTION constexpr auto operator-(First &&first, Second &&second) {
                return plus(std::forward<First>(first), -std::forward<Second>(second));
            }

        } // namespace arithmetic
    }     // namespace int_vector
} // namespace gridtools
