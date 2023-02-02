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

#include <tuple>
#include <type_traits>

#include "../common/defs.hpp"
#include "../common/for_each.hpp"
#include "../common/hymap.hpp"
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/unknown_kind.hpp"
#include "data_store.hpp"
#include "traits.hpp"

namespace gridtools {
    namespace storage {
        namespace builder_impl_ {
            namespace param {
                struct type {};
                struct lengths {};
                struct id {};
                struct name {};
                struct halos {};
                struct initializer {};
                struct layout {};
            } // namespace param

            template <class T>
            integral_constant<int_t, T::value> normalize_dimension(T, std::true_type) {
                return {};
            }

            template <class T>
            int_t normalize_dimension(T const &obj, std::false_type) {
                return obj;
            }

            template <class Layout, class Info>
            auto restore_indices(Info const &info, Layout, int src) {
                auto l = info.lengths();
                auto s = info.strides();
                array<int, Info::ndims> res;
                for (size_t i = 0; i < Info::ndims; ++i) {
                    auto n = Layout::at(i);
                    if (n == -1)
                        res[i] = l[i] - 1;
                    else if (n == 0)
                        res[i] = src / s[i];
                    else
                        res[i] = src % s[Layout::find(n - 1)] / s[i];
                }
                return res;
            }

            template <class Fun, class T, class Layout, class Info, size_t... Is>
            void initializer_impl(Fun const &fun, T *dst, Layout layout, Info const &info, std::index_sequence<Is...>) {
                int length = info.length();
                auto in_range = [&](auto const &indices) {
                    for (auto ok : {(tuple_util::get<Is>(indices) < tuple_util::get<Is>(info.native_lengths()))...})
                        if (!ok)
                            return false;
                    return true;
                };
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (int i = 0; i < length; ++i) {
                    auto indices = restore_indices(info, layout, i);
                    if (in_range(indices))
                        dst[i] = fun(tuple_util::get<Is>(indices)...);
                }
            }

            template <class Fun>
            auto wrap_initializer(Fun fun) {
                return [fun = std::move(fun)](auto *dst, auto layout, auto const &info) {
                    initializer_impl(
                        fun, dst, layout, info, std::make_index_sequence<std::decay_t<decltype(info)>::ndims>());
                };
            }

            template <class T>
            auto wrap_value(T const &value) {
                return [value = std::move(value)](auto *dst, auto, auto const &info) {
                    int length = info.length();
#ifdef _OPENMP
#pragma omp parallel for
#endif
                    for (int i = 0; i < length; ++i)
                        dst[i] = value;
                };
            }

            template <class Traits, class Layout>
            struct custom_traits : Traits {
                friend Layout storage_layout(custom_traits, std::integral_constant<size_t, Layout::masked_length>) {
                    return {};
                }
            };

            template <class... Keys>
            struct keys {
                template <class... Vals>
                struct values {
                    static_assert(sizeof...(Vals) == sizeof...(Keys), "invalid hymap");

                    std::tuple<Vals...> m_vals;

                    template <class Arg,
                        class... Args,
                        std::enable_if_t<std::is_constructible<std::tuple<Vals...>, Arg &&, Args &&...>::value, int> =
                            0>
                    constexpr values(Arg &&arg, Args &&...args) noexcept
                        : m_vals{std::forward<Arg>(arg), std::forward<Args>(args)...} {}

                    values() = default;
                    values(values const &) = default;
                    values(values &&) = default;
                    values &operator=(values const &) = default;
                    values &operator=(values &&) = default;

                    struct values_tuple_util_getter {
                        template <size_t I>
                        static constexpr decltype(auto) get(values const &obj) {
                            return tuple_util::get<I>(obj.m_vals);
                        }
                        template <size_t I>
                        static decltype(auto) get(values &obj) {
                            return tuple_util::get<I>(obj.m_vals);
                        }
                        template <size_t I>
                        static constexpr decltype(auto) get(values &&obj) {
                            return tuple_util::get<I>(std::move(obj).m_vals);
                        }
                    };
                    friend values_tuple_util_getter tuple_getter(values const &) { return {}; }
                    friend keys hymap_get_keys(values const &) { return {}; }
                };
            };

            template <class, bool... Masks>
            struct apply_mask;

            template <int... Args, bool... Masks>
            struct apply_mask<layout_map<Args...>, Masks...> {
                static constexpr int correction(int arg) {
                    int res = 0;
                    for (int i : {(!Masks && Args >= 0 && Args < arg ? 1 : 0)...})
                        res += i;
                    return res;
                }

                using type = layout_map<(Masks ? Args - correction(Args) : -1)...>;
            };

            template <class Traits, class Params>
            class builder_type {
                Params m_params;

                template <class Key>
                using has = has_key<Params, Key>;

                template <class Key>
                using value_type = std::decay_t<decltype(at_key_with_default<Key, void>(std::declval<Params>()))>;

                template <class Key>
                decltype(auto) value() const {
                    return at_key<Key>(m_params);
                }

                template <class Key, class Default>
                decltype(auto) value() const {
                    return at_key_with_default<Key, std::decay_t<Default>>(m_params);
                }

                template <class Key, class Value, std::enable_if_t<!has<Key>::value, int> = 0>
                auto add_value(Value value) const {
                    auto params = hymap::concat(m_params, typename keys<Key>::template values<Value>(std::move(value)));
                    return builder_type<Traits, decltype(params)>{std::move(params)};
                }

                template <class Key, class Value, std::enable_if_t<has<Key>::value, int> = 0>
                auto add_value(Value) const {
                    return *this;
                }

                template <class Key, class Type>
                auto add_type() const {
                    return add_value<Key>(Type());
                }

                template <size_t N>
                static void check_dimensions_number(param::lengths) {
                    static_assert(N == tuple_util::size<value_type<param::lengths>>::value,
                        "number of args is inconsistent with builder.dimensions(...);");
                }
                template <size_t N>
                static void check_dimensions_number(param::halos) {
                    static_assert(N == tuple_util::size<value_type<param::halos>>::value,
                        "number of args is inconsistent with builder.halos(...);");
                }
                template <size_t N>
                static void check_dimensions_number(param::layout) {
                    static_assert(N == value_type<param::layout>::masked_length,
                        "number of args is inconsistent with builder.layout<...>() or builder.selector<...>();");
                }
                template <size_t N>
                static void check_dimensions_number() {
                    static_assert(N, "number of dimensions should be positive");
                    for_each<meta::filter<has, meta::list<param::lengths, param::halos, param::layout>>>(
                        [](auto param) { check_dimensions_number<N>(param); });
                }

                constexpr builder_type(Params params) : m_params(std::move(params)) {}

                template <class, class>
                friend class builder_type;

              public:
                template <size_t Size = tuple_util::size<Params>::value, std::enable_if_t<Size == 0, int> = 0>
                constexpr builder_type() {}

                template <class T>
                auto type() const {
                    static_assert(!has<param::type>::value, "storage type is set twice");
                    return add_type<param::type, meta::lazy::id<T>>();
                }

                template <int I>
                auto id() const {
                    static_assert(!has<param::id>::value, "storage id or unknown_id is set twice");
                    return add_type<param::id, std::integral_constant<int, I>>();
                }

                auto unknown_id() const {
                    static_assert(!has<param::id>::value, "storage id or unknown_id is set twice");
                    return add_type<param::id, sid::unknown_kind>();
                }

                template <int... Args>
                auto layout() const {
                    static_assert(!has<param::layout>::value, "storage layout/selector is set twice");
                    check_dimensions_number<sizeof...(Args)>();
                    return add_type<param::layout, layout_map<Args...>>();
                }

                template <bool... Args>
                auto selector() const {
                    static_assert(!has<param::layout>::value, "storage layout/selector is set twice");
                    check_dimensions_number<sizeof...(Args)>();
                    using layout_t = typename apply_mask<traits::layout_type<Traits, sizeof...(Args)>, Args...>::type;
                    return add_type<param::layout, layout_t>();
                }

                auto name(std::string value) const {
                    static_assert(!has<param::name>::value, "storage name is set twice");
                    return add_value<param::name>(std::move(value));
                }

                template <class... Args>
                auto dimensions(Args const &...values) const {
                    static_assert(!has<param::lengths>::value, "storage dimensions are set twice");
                    static_assert(std::conjunction<std::is_convertible<Args const &, uint_t>...>::value,
                        "builder.dimensions(...) arguments should be convertible to unsigned int");
                    check_dimensions_number<sizeof...(Args)>();
                    return add_value<param::lengths>(
                        tuple(normalize_dimension(values, is_integral_constant<Args>())...));
                }

                template <class... Args>
                auto halos(Args const &...values) const {
                    static_assert(!has<param::halos>::value, "storage dimensions are set twice");
                    static_assert(std::conjunction<std::is_convertible<Args const &, int>...>::value,
                        "builder.halos(...) arguments should be convertible to int");
                    check_dimensions_number<sizeof...(Args)>();
                    return add_value<param::halos>(array<int, sizeof...(Args)>{static_cast<int>(values)...});
                }

                template <class Fun>
                auto initializer(Fun fun) const {
                    static_assert(!has<param::initializer>::value, "storage initializer/value is set twice");
                    return add_value<param::initializer>(wrap_initializer(std::move(fun)));
                }

                template <class T>
                auto value(T value) const {
                    static_assert(!has<param::initializer>::value, "storage initializer/value is set twice");
                    return add_value<param::initializer>(wrap_value(std::move(value)));
                }

                auto build() const {
                    static_assert(has<param::type>::value, "storage type is not set");
                    static_assert(has<param::lengths>::value, "storage lengths are not set");
                    using traits_t =
                        meta::if_c<has<param::layout>::value, custom_traits<Traits, value_type<param::layout>>, Traits>;
                    auto &&lengths = value<param::lengths>();
                    auto &&name = value<param::name, std::string>();
                    constexpr auto n = tuple_util::size<decltype(lengths)>::value;
                    auto &&halos = value<param::halos, array<int, n>>();
                    auto initializer = value<param::initializer, uninitialized>();
                    return make_data_store<traits_t, typename value_type<param::type>::type, value_type<param::id>>(
                        name, lengths, halos, initializer);
                }

                auto operator()() const { return build(); }
            };
            template <class Traits>
            constexpr builder_type<Traits, keys<>::values<>> builder = {};
        } // namespace builder_impl_
        using builder_impl_::builder;
    } // namespace storage
} // namespace gridtools
