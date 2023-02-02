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

#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../meta/at.hpp"
#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {

    namespace impl_ {
        template <size_t I, class T, bool = std::is_empty_v<T>>
        struct tuple_leaf {
            T m_value;

            tuple_leaf(tuple_leaf const &) = default;
            tuple_leaf(tuple_leaf &&) = default;
            tuple_leaf &operator=(tuple_leaf const &) = default;
            tuple_leaf &operator=(tuple_leaf &&) = default;

            constexpr GT_FUNCTION tuple_leaf() noexcept : m_value() {}

            template <class Arg, std::enable_if_t<std::is_constructible_v<T, Arg &&>, int> = 0>
            constexpr GT_FUNCTION tuple_leaf(Arg &&arg) noexcept : m_value(std::forward<Arg>(arg)) {}
        };

        template <size_t I, class T>
        struct tuple_leaf<I, T, true> : T {
            tuple_leaf() = default;
            tuple_leaf(tuple_leaf const &) = default;
            tuple_leaf(tuple_leaf &&) = default;
            tuple_leaf &operator=(tuple_leaf const &) = default;
            tuple_leaf &operator=(tuple_leaf &&) = default;

            template <class Arg, std::enable_if_t<std::is_constructible_v<T, Arg &&>, int> = 0>
            constexpr GT_FUNCTION tuple_leaf(Arg &&arg) noexcept : T(std::forward<Arg>(arg)) {}
        };

        struct tuple_leaf_getter {
            template <size_t I, class T>
            static constexpr GT_FUNCTION T const &get(tuple_leaf<I, T, false> const &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &get(tuple_leaf<I, T, false> &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &&get(tuple_leaf<I, T, false> &&obj) noexcept {
                return static_cast<T &&>(get<I>(obj));
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T const &get(tuple_leaf<I, T, true> const &obj) noexcept {
                return obj;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &get(tuple_leaf<I, T, true> &obj) noexcept {
                return obj;
            }

            template <size_t I, class T>
            static constexpr GT_FUNCTION T &&get(tuple_leaf<I, T, true> &&obj) noexcept {
                return static_cast<T &&>(obj);
            }
        };

        template <class Indices, class... Ts>
        struct tuple_impl;

        template <size_t... Is, class... Ts>
        struct tuple_impl<std::index_sequence<Is...>, Ts...> : tuple_leaf<Is, Ts>... {
            tuple_impl() = default;
            tuple_impl(tuple_impl const &) = default;
            tuple_impl(tuple_impl &&) = default;
            tuple_impl &operator=(tuple_impl const &) = default;
            tuple_impl &operator=(tuple_impl &&) = default;

            template <class... Args>
            constexpr GT_FUNCTION tuple_impl(Args &&...args) noexcept
                : tuple_leaf<Is, Ts>(std::forward<Args>(args))... {}

            template <class Src>
            constexpr GT_FUNCTION tuple_impl(Src &&src) noexcept
                : tuple_leaf<Is, Ts>(tuple_leaf_getter::get<Is>(std::forward<Src>(src)))... {}

            constexpr GT_FORCE_INLINE void swap(tuple_impl &other) noexcept {
                using std::swap;
                (..., swap(tuple_leaf_getter::get<Is>(*this), tuple_leaf_getter::get<Is>(other)));
            }

            template <class... Args,
                std::enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                     std::conjunction_v<std::is_assignable<Ts &, Args const &>...>,
                    int> = 0>
            constexpr GT_FUNCTION void assign(tuple_impl<std::index_sequence<Is...>, Args...> const &src) noexcept {
                (..., (tuple_leaf_getter::get<Is>(*this) = tuple_leaf_getter::get<Is>(src)));
            }

            template <class... Args,
                std::enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                     std::conjunction_v<std::is_assignable<Ts &, Args &&>...>,
                    int> = 0>
            constexpr GT_FUNCTION void assign(tuple_impl<std::index_sequence<Is...>, Args...> &&src) noexcept {
                (..., (tuple_leaf_getter::get<Is>(*this) = tuple_leaf_getter::get<Is>(std::move(src))));
            }
        };
    } // namespace impl_

    /**
     *  Simplified host/device aware implementation of std::tuple interface.
     *
     *  Nuances
     *  =======
     *
     *  - get/tuple_element/tuple_size, comparision operators etc. are not implemented. Instead `tuple` is adopted
     *    to use with tuple_util library.
     *  - `allocator` aware constructors are not implemented
     *  - all constructors are implicit. [which violates the Standard]
     *  - element wise direct constructor is not sfinae friendly
     *  - all methods declared as noexcept [which violates the Standard]
     *  - `swap` is implemented as a `__host__` function because it can call `std::swap`
     *
     */
    template <class... Ts>
    class tuple {
        impl_::tuple_impl<std::index_sequence_for<Ts...>, Ts...> m_impl;

        struct getter {
            template <size_t I>
            static constexpr GT_FUNCTION decltype(auto) get(tuple const &obj) noexcept {
                return impl_::tuple_leaf_getter::get<I>(obj.m_impl);
            }

            template <size_t I>
            static constexpr GT_FUNCTION decltype(auto) get(tuple &obj) noexcept {
                return impl_::tuple_leaf_getter::get<I>(obj.m_impl);
            }

            template <size_t I>
            static constexpr GT_FUNCTION decltype(auto) get(tuple &&obj) noexcept {
                return impl_::tuple_leaf_getter::get<I>(std::move(obj.m_impl));
            }
        };
        friend getter tuple_getter(tuple const &) { return {}; }

        template <class...>
        friend class tuple;

      public:
        tuple() = default;
        tuple(tuple const &) = default;
        tuple(tuple &&) = default;
        tuple &operator=(tuple const &) = default;
        tuple &operator=(tuple &&) = default;

        constexpr GT_FUNCTION tuple(Ts const &...args) noexcept : m_impl(args...) {}

        template <class... Args,
            std::enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                 std::conjunction_v<std::is_constructible<Ts, Args &&>...>,
                int> = 0>
        constexpr GT_FUNCTION tuple(Args &&...args) noexcept : m_impl(std::forward<Args>(args)...) {}

        template <class... Args,
            std::enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                 std::conjunction_v<std::is_constructible<Ts, Args const &>...>,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Args...> const &src) noexcept : m_impl(src.m_impl) {}

        template <class... Args,
            std::enable_if_t<sizeof...(Ts) == sizeof...(Args) &&
                                 std::conjunction_v<std::is_constructible<Ts, Args &&>...>,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Args...> &&src) noexcept : m_impl(std::move(src).m_impl) {}

        GT_FORCE_INLINE void swap(tuple &other) noexcept { m_impl.swap(other.m_impl); }

        template <class Other>
        constexpr GT_FUNCTION tuple &operator=(Other &&other) {
            m_impl.assign(std::forward<Other>(other).m_impl);
            return *this;
        }
    };

    template <class T>
    class tuple<T> {
        T m_value;
        struct getter {
            template <size_t I, std::enable_if_t<I == 0, int> = 0>
            static constexpr GT_FUNCTION T const &get(tuple const &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, std::enable_if_t<I == 0, int> = 0>
            static constexpr GT_FUNCTION T &get(tuple &obj) noexcept {
                return obj.m_value;
            }

            template <size_t I, std::enable_if_t<I == 0, int> = 0>
            static constexpr GT_FUNCTION T &&get(tuple &&obj) noexcept {
                return static_cast<T &&>(obj.m_value);
            }
        };
        friend getter tuple_getter(tuple const &) { return {}; }

        template <class...>
        friend class tuple;

      public:
        constexpr GT_FUNCTION tuple() noexcept : m_value() {}

        tuple(tuple const &) = default;
        tuple(tuple &&) = default;
        tuple &operator=(tuple const &) = default;
        tuple &operator=(tuple &&) = default;

        constexpr GT_FUNCTION tuple(T const &arg) noexcept : m_value(arg) {}

        template <class Arg, std::enable_if_t<std::is_constructible_v<T, Arg &&>, int> = 0>
        constexpr GT_FUNCTION tuple(Arg &&arg) noexcept : m_value(std::forward<Arg>(arg)) {}

        template <class Arg,
            std::enable_if_t<std::is_constructible_v<T, Arg const &> && !std::is_convertible_v<tuple<Arg> const &, T> &&
                                 !std::is_constructible_v<T, tuple<Arg> const &> && !std::is_same_v<T, Arg>,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Arg> const &src) noexcept : m_value(src.m_value) {}

        template <class Arg,
            std::enable_if_t<std::is_constructible_v<T, Arg &&> && !std::is_convertible_v<tuple<Arg>, T> &&
                                 !std::is_constructible_v<T, tuple<Arg>> && !std::is_same_v<T, Arg>,
                int> = 0>
        constexpr GT_FUNCTION tuple(tuple<Arg> &&src) noexcept : m_value(std::move(src).m_value) {}

        GT_FORCE_INLINE void swap(tuple &other) noexcept {
            using std::swap;
            swap(m_value, other.m_value);
        }

        template <class Arg, std::enable_if_t<std::is_assignable_v<T &, Arg const &>, int> = 0>
        constexpr GT_FUNCTION tuple &operator=(tuple<Arg> const &src) noexcept {
            m_value = src.m_value;
            return *this;
        }

        template <class Arg, std::enable_if_t<std::is_assignable_v<T &, Arg &&>, int> = 0>
        constexpr GT_FUNCTION tuple &operator=(tuple<Arg> &&src) noexcept {
            m_value = std::move(src).m_value;
            return *this;
        }
    };

    template <>
    class tuple<> {
        friend tuple tuple_getter(tuple const &) { return {}; }

      public:
        constexpr GT_FORCE_INLINE void swap(tuple &) noexcept {}
    };

    template <class... Ts>
    constexpr GT_FORCE_INLINE void swap(tuple<Ts...> &lhs, tuple<Ts...> &rhs) noexcept {
        lhs.swap(rhs);
    }

    template <size_t I, class... Ts>
    constexpr GT_FUNCTION decltype(auto) get(tuple<Ts...> const &obj) noexcept {
        return decltype(tuple_getter(obj))::template get<I>(obj);
    }

    template <size_t I, class... Ts>
    constexpr GT_FUNCTION decltype(auto) get(tuple<Ts...> &obj) noexcept {
        return decltype(tuple_getter(std::declval<tuple<Ts...> const &>()))::template get<I>(obj);
    }

    template <size_t I, class... Ts>
    constexpr GT_FUNCTION decltype(auto) get(tuple<Ts...> &&obj) noexcept {
        return decltype(tuple_getter(std::declval<tuple<Ts...> const &>()))::template get<I>(std::move(obj));
    }
} // namespace gridtools

namespace std {
    template <class... Ts>
    struct tuple_size<::gridtools::tuple<Ts...>> : integral_constant<size_t, sizeof...(Ts)> {};

    template <size_t I, class... Ts>
    struct tuple_element<I, ::gridtools::tuple<Ts...>> {
        using type = gridtools::meta::at_c<::gridtools::tuple<Ts...>, I>;
    };
} // namespace std
