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

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup pair Simple Pair
        @{
    */

    /**
       @brief simple pair with constexpr constructor

       NOTE: can be replaced by std::pair
     */
    template <typename T1, typename T2>
    struct pair {
        pair() = default;

        template <class U1, class U2>
        constexpr GT_FUNCTION pair(const std::pair<U1, U2> &p) : pair(p.first, p.second) {}

        template <class U1, class U2>
        constexpr GT_FUNCTION pair(std::pair<U1, U2> &&p) : pair(std::move(p.first), std::move(p.second)) {}

        constexpr GT_FUNCTION pair(T1 const &t1_, T2 const &t2_) : first(t1_), second(t2_) {}

        template <class U1, class U2>
        constexpr GT_FUNCTION pair(U1 &&t1_, U2 &&t2_) : first(std::forward<U1>(t1_)), second(std::forward<U2>(t2_)) {}

        template <class U1, class U2, std::enable_if_t<!std::is_same_v<pair<U1, U2>, pair>, int> = 0>
        constexpr GT_FUNCTION pair(pair<U1, U2> const &p) : first(p.first), second(p.second) {}

        template <class U1, class U2, std::enable_if_t<!std::is_same_v<pair<U1, U2>, pair>, int> = 0>
        constexpr GT_FUNCTION pair(pair<U1, U2> &&p) : first(std::move(p.first)), second(std::move(p.second)) {}

        template <typename U1, typename U2, std::enable_if_t<!std::is_same_v<pair<U1, U2>, pair>, int> = 0>
        GT_FUNCTION pair &operator=(const pair<U1, U2> &other) {
            first = other.first;
            second = other.second;
            return *this;
        }

        template <typename U1, typename U2, std::enable_if_t<!std::is_same_v<pair<U1, U2>, pair>, int> = 0>
        GT_FUNCTION pair &operator=(pair<U1, U2> &&other) noexcept {
            first = std::move(other.first);
            second = std::move(other.second);
            return *this;
        }

        T1 first;
        T2 second;
    };

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator==(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator!=(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return !(lhs == rhs);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator<(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return lhs.first < rhs.first || (!(rhs.first < lhs.first) && lhs.second < rhs.second);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator>(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return rhs < lhs;
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator<=(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return !(rhs < lhs);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator>=(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return !(lhs < rhs);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION pair<T1, T2> make_pair(T1 const &t1_, T2 const &t2_) {
        return pair<T1, T2>(t1_, t2_);
    }
    /** @} */
    /** @} */

    namespace pair_impl_ {
        template <std::size_t I>
        struct pair_get;

        template <>
        struct pair_get<0> {
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION const T1 &const_get(const pair<T1, T2> &p) noexcept {
                return p.first;
            }
            template <typename T1, typename T2>
            static GT_FUNCTION T1 &get(pair<T1, T2> &p) noexcept {
                return p.first;
            }
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION T1 &&move_get(pair<T1, T2> &&p) noexcept {
                return std::move(p.first);
            }
        };
        template <>
        struct pair_get<1> {
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION const T2 &const_get(const pair<T1, T2> &p) noexcept {
                return p.second;
            }
            template <typename T1, typename T2>
            static GT_FUNCTION T2 &get(pair<T1, T2> &p) noexcept {
                return p.second;
            }
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION T2 &&move_get(pair<T1, T2> &&p) noexcept {
                return std::move(p.second);
            }
        };

        struct getter {
            template <std::size_t I, class T1, class T2>
            static constexpr GT_FUNCTION decltype(auto) get(pair<T1, T2> &p) noexcept {
                return pair_get<I>::get(p);
            }

            template <std::size_t I, class T1, class T2>
            static constexpr GT_FUNCTION decltype(auto) get(const pair<T1, T2> &p) noexcept {
                return pair_get<I>::const_get(p);
            }

            template <std::size_t I, class T1, class T2>
            static constexpr GT_FUNCTION decltype(auto) get(pair<T1, T2> &&p) noexcept {
                return pair_get<I>::move_get(std::move(p));
            }
        };
    } // namespace pair_impl_

    template <std::size_t I, class T1, class T2>
    GT_FUNCTION decltype(auto) get(pair<T1, T2> &p) noexcept {
        return pair_impl_::pair_get<I>::get(p);
    }

    template <std::size_t I, class T1, class T2>
    constexpr GT_FUNCTION decltype(auto) get(const pair<T1, T2> &p) noexcept {
        return pair_impl_::pair_get<I>::const_get(p);
    }

    template <std::size_t I, class T1, class T2>
    constexpr GT_FUNCTION decltype(auto) get(pair<T1, T2> &&p) noexcept {
        return pair_impl_::pair_get<I>::move_get(std::move(p));
    }

    template <class T1, class T2>
    pair_impl_::getter tuple_getter(pair<T1, T2> const &);
} // namespace gridtools

namespace std {
    template <class T1, class T2>
    struct tuple_size<::gridtools::pair<T1, T2>> : integral_constant<size_t, 2> {};

    template <class T1, class T2>
    struct tuple_element<0, ::gridtools::pair<T1, T2>> {
        using type = T1;
    };
    template <class T1, class T2>
    struct tuple_element<1, ::gridtools::pair<T1, T2>> {
        using type = T2;
    };
} // namespace std
