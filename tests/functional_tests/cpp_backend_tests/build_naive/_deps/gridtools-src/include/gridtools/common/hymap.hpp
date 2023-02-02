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
 *  @file
 *
 *  Hybrid Map (aka hymap) Concept Definition
 *  -----------------------------------------
 *  Hymap is a `tuple_like` (see tuple_util.hpp for definition) that additionally holds a type list of keys.
 *  This type list should be a set. The existence of it allows to use an alternative syntax for element accessor
 *  (in addition to `tuple_util::get`). Say you have a hymap `obj` that is a `tuple_like` of two elements and has
 *  keys `a` and `b`. To access the elements you can use: `at_key<a>(obj)` and `at_key<b>(obj)` which are semantically
 *  the same as `get<0>(obj)` and `get<1>(obj)`.
 *
 *  Regarding the Naming
 *  --------------------
 *  Hymaps provides a mechanism for mapping compile time keys to run time values. That is why it is hybrid map.
 *  Keys exist only in compile time -- it is enough to have just a type list to keep them; values are kept in the
 *  `tuple_like` -- that is why hymap also models `tuple_like` for its values.
 *
 *  Concept modeler API
 *  -------------------
 *  Hymaps provide the mentioned keys type lists by declaring the function that should be available by ADL:
 *  `Keys hymap_get_keys(Hymap)`.
 *
 *  Another ADL searchable function is provided to specify the way how to create hymap class from keys and values:
 *   - `FromKeysValuesMataClass hymap_from_keys_values(Hymap)`;
 *
 *  The Default Behaviour
 *  ---------------------
 *  If hymap doesn't provide `hymap_get_keys`, the default is taken which is:
 *  `meta::list<integral_constant<int, 0>, ..., integral_constant<int, N> >` where N is `tuple_util::size` of hymap.
 *  This means that any `tuple_like` automatically models Hymap. And for the plain `tuple_like`'s you can use
 *  `at_key<integral_constant<int, N>>(obj)` instead of `tuple_util::get<N>(obj)`.
 *
 *  `hymap_from_keys_values` has a default only for the for hymaps that are structured like:
 *     `SomeKeyTemplate<key_a, key_b, key_c>::values<val_a, val_b, val_c>`
 *
 *  User API
 *  --------
 *  Run time: a single function `target_name::at_key<Key>(hymap_obj)` is provided where `target_name` is `host`,
 * `device` or `host_device`. `at_key` without `target_name` is an alias of `host::at_key`.
 *
 *  Compile time:
 *  - `get_keys` metafunction. Usage: `get_keys<Hymap>`
 *  - `has_key` metafunction. Usage: `has_key<Hymap, Key>`
 *  - `get_from_keys_values` metafunction. It returns a meta class that serves as a meta constructor from <Keys, Values>
 *     Usage:
 *     ```
 *     // Say Hymap class is hymap that maps <a, b> types to the values of <int, double>.
 *     // And we want to construct another instantiation of that kind of Hymap (Hymap2) that
 *     // maps <a, b, c> to <int*, int**, int**> values.
 *
 *     // First we extract meta ctor from Hymap using get_from_keys_values
 *     using ctor = get_from_keys_values<Hymap>;
 *
 *     // Next we apply this meta ctor with desired types
 *     using Hymap2 = ctor::template apply<list<a, b, c>, list<int*, int**, int***>>;
 *     ```
 *
 *  TODO(anstaf): add usage examples here
 *
 *  Gridtools implementation of Hymap
 *  ---------------------------------
 *
 *  Usage
 *  -----
 *  ```
 *    struct a;
 *    struct b;
 *    using my_map_t = hymap::keys<a, b>::values<int, double>;
 *  ```
 *
 *  Composing with `tuple_util` library
 *  -----------------------------------
 *  Because `hymap` is also a `tuple_like`, all `tuple_util` stuff is at your service.
 *  For example:
 *  - transforming the values of the hymap:
 *    `auto dst_hymap = tuple_util::transform(change_value_functor, src_hymap);`
 *  - making a map:
 *    `auto a_map = hymap::keys<a, b>::values('a', 42)`
 *  - converting to a map:
 *    `auto a_map = tuple_util::convert_to<hymap::keys<a, b>::values>(a_tuple_with_values)`
 */

#ifndef GT_TARGET_ITERATING
//// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_HYMAP_HPP_
#define GT_COMMON_HYMAP_HPP_

#include <type_traits>
#include <utility>

#include "../meta.hpp"
#include "defs.hpp"
#include "host_device.hpp"
#include "integral_constant.hpp"
#include "tuple.hpp"
#include "tuple_util.hpp"

namespace gridtools {

    namespace hymap_impl_ {

        template <class I>
        using get_key = integral_constant<int_t, I::value>;

        template <class Tup, class Ts = tuple_util::traits::to_types<Tup>>
        using default_keys = meta::transform<get_key, meta::make_indices_for<Ts>>;

        struct not_provided;

        not_provided hymap_get_keys(...);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        std::enable_if_t<!std::is_same_v<Res, not_provided>, Res> get_keys_fun(T const &);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        std::enable_if_t<std::is_same_v<Res, not_provided>, default_keys<T>> get_keys_fun(T const &);

        template <class T>
        using get_keys = decltype(::gridtools::hymap_impl_::get_keys_fun(std::declval<T const &>()));

        template <class Map, class = void>
        struct values_are_nested_in_keys : std::false_type {};

        template <template <class...> class L, class... Values>
        struct values_are_nested_in_keys<L<Values...>,
            std::enable_if_t<std::is_same_v<L<Values...>, typename get_keys<L<Values...>>::template values<Values...>>>>
            : std::true_type {};

        template <template <class...> class Ctor>
        struct from_key_values_nested {
            template <class Keys, class Values>
            using apply = meta::rename<meta::rename<Ctor, Keys>::template values, Values>;
        };

        template <class Map>
        using default_from_keys_values = from_key_values_nested<meta::ctor<get_keys<Map>>::template apply>;

        not_provided hymap_from_keys_values(...);

        template <class T, class Res = decltype(hymap_from_keys_values(std::declval<T const &>()))>
        std::enable_if_t<!std::is_same_v<Res, not_provided>, Res> get_from_keys_values_fun(T const &);

        template <class T, class Res = decltype(hymap_from_keys_values(std::declval<T const &>()))>
        std::enable_if_t<std::is_same_v<Res, not_provided> && values_are_nested_in_keys<T>::value,
            default_from_keys_values<T>>
        get_from_keys_values_fun(T const &);

        template <class T>
        using get_from_keys_values =
            decltype(::gridtools::hymap_impl_::get_from_keys_values_fun(std::declval<T const &>()));

        template <class Key, class Map>
        using element_at = tuple_util::element<meta::st_position<get_keys<Map>, Key>::value, Map>;

        template <class T, class Keys = get_keys<T>>
        struct keys_are_legit_sfinae
            : std::bool_constant<meta::is_set<Keys>::value && meta::length<Keys>::value == tuple_util::size<T>::value> {
        };

        template <class, class = void>
        struct keys_are_legit : std::false_type {};

        template <class T>
        struct keys_are_legit<T, std::enable_if_t<keys_are_legit_sfinae<T>::value>> : std::true_type {};

    } // namespace hymap_impl_

    template <class T>
    using is_hymap = std::bool_constant<is_tuple_like<T>::value && hymap_impl_::keys_are_legit<T>::value>;

#ifdef __cpp_concepts
    namespace concepts {
        template <class T>
        concept hymap = is_hymap<T>::value;
    }
#endif

    using hymap_impl_::element_at;
    using hymap_impl_::get_from_keys_values;
    using hymap_impl_::get_keys;

    template <class Map, class Key>
    using has_key = meta::st_contains<hymap_impl_::get_keys<Map>, Key>;

    namespace hymap {
        template <class...>
        struct keys {
            template <class...>
            struct values;

#if !defined(__NVCC__) && defined(__clang__) && __clang_major__ <= 15
            template <class... Vs>
            values(Vs const &...) -> values<Vs...>;
#endif

            // NVCC 11 fails to do class template deduction in the case of nested templates
            template <class... Args>
            static constexpr GT_FUNCTION values<Args...> make_values(Args const &...args) {
                return {args...};
            }
        };

        template <class... Keys>
        template <class... Vals>
        struct keys<Keys...>::values {
            static_assert(sizeof...(Vals) == sizeof...(Keys), "invalid hymap");

            tuple<Vals...> m_vals;

            template <class... Args,
                std::enable_if_t<std::conjunction_v<std::is_constructible<Vals, Args>...>, int> = 0>
            constexpr GT_FUNCTION values(Args &&...args) noexcept : m_vals{std::forward<Args>(args)...} {}

            constexpr GT_FUNCTION values(Vals const &...args) noexcept : m_vals(args...) {}

            constexpr GT_FUNCTION values(tuple<Vals...> &&args) noexcept : m_vals(std::move(args)) {}
            constexpr GT_FUNCTION values(tuple<Vals...> const &args) noexcept : m_vals(args) {}

            values() = default;
            values(values const &) = default;
            values(values &&) = default;
            values &operator=(values const &) = default;
            values &operator=(values &&) = default;

            template <class Src>
            constexpr GT_FUNCTION
                std::enable_if_t<((!std::is_same_v<values, std::decay_t<Src>> && is_hymap<std::decay_t<Src>>::value) &&
                                     ... &&
                                     std::is_assignable_v<Vals &,
                                         std::add_lvalue_reference_t<element_at<Keys, std::remove_reference_t<Src>>>>),
                    values &>
                operator=(Src &&src) {
                (...,
                    (tuple_util::host_device::get<meta::st_position<meta::list<Keys...>, Keys>::value>(m_vals) =
                            tuple_util::host_device::get<meta::st_position<get_keys<std::decay_t<Src>>, Keys>::value>(
                                src)));
                return *this;
            }

            GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(values, m_vals);

            friend keys hymap_get_keys(values const &) { return {}; }
        };

        template <>
        template <>
        struct keys<>::values<> {
            friend values tuple_getter(values const &) { return {}; }
            friend keys hymap_get_keys(values const &) { return {}; }
        };

        template <class HyMap>
        using to_meta_map = meta::zip<get_keys<HyMap>, tuple_util::traits::to_types<HyMap>>;

        template <class Keys, class Values, template <class...> class KeyCtor = keys>
        using from_keys_values = meta::rename<meta::rename<KeyCtor, Keys>::template values, Values>;

        template <class MetaMap,
            template <class...> class KeyCtor = keys,
            class KeysAndValues =
                meta::if_<meta::is_empty<MetaMap>, meta::list<meta::list<>, meta::list<>>, meta::transpose<MetaMap>>>
        using from_meta_map = from_keys_values<meta::first<KeysAndValues>, meta::second<KeysAndValues>, KeyCtor>;

        namespace impl_ {
            template <class Maps>
            using merged_keys = meta::dedup<meta::transform<meta::first, meta::flatten<Maps>>>;

            template <class Key>
            struct find_f {
                template <class Map>
                using apply = meta::second<meta::mp_find<Map, Key, meta::list<void, void>>>;
            };

            template <class State, class Val>
            using get_first_folder = meta::if_<std::is_void<State>, Val, State>;

            template <class Maps>
            struct merged_value_f {
                template <class Key>
                using apply = meta::foldl<get_first_folder, void, meta::transform<find_f<Key>::template apply, Maps>>;
            };

            template <class Src>
            using map_of_refs = decltype(tuple_util::transform(identity{}, std::declval<Src>()));

            template <class Maps,
                class RefMaps = meta::transform<map_of_refs, Maps>,
                class MetaMaps = meta::transform<to_meta_map, RefMaps>,
                class Keys = merged_keys<MetaMaps>,
                class Values = meta::transform<merged_value_f<MetaMaps>::template apply, Keys>>
            using merged_old = from_keys_values<Keys, Values>;

            struct concat_result_maker_f {
                template <class Values, class Maps, class Keys = meta::flatten<meta::transform<get_keys, Maps>>>
                using apply = typename get_from_keys_values<meta::first<Maps>>::template apply<Keys, Values>;
            };
        } // namespace impl_
    }     // namespace hymap
} // namespace gridtools

#define GT_FILENAME <gridtools/common/hymap.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_HYMAP_HPP_
#else  // GT_TARGET_ITERATING

namespace gridtools {
    GT_TARGET_NAMESPACE {
        template <class Key,
            class Map,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value != tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) at_key(Map && map) noexcept {
            return tuple_util::GT_TARGET_NAMESPACE_NAME::get<I::value>(std::forward<Map>(map));
        }

        template <class Key,
            class Map,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value == tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET void at_key(Map &&) noexcept {
            static_assert(sizeof(Key) != sizeof(Key), "wrong key");
        }

        template <class Key,
            class Default,
            class Map,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value != tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) at_key_with_default(Map && map) noexcept {
            return tuple_util::GT_TARGET_NAMESPACE_NAME::get<I::value>(std::forward<Map>(map));
        }

        template <class Key,
            class Default,
            class Map,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value == tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE constexpr Default at_key_with_default(Map &&) noexcept {
            return {};
        }
    }

    namespace hymap {
        GT_TARGET_NAMESPACE {
            namespace hymap_detail {
                template <class Fun, class Keys>
                struct adapter_f {
                    Fun m_fun;
                    template <size_t I, class Value, class Key = meta::at_c<Keys, I>>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(Value &&value) const {
                        return m_fun.template operator()<Key>(std::forward<Value>(value));
                    }
                };

                template <class Key>
                struct at_generator_f {
                    template <class Value>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(Value &&value) const {
                        return at_key<Key>(std::forward<Value>(value));
                    }
                };
            } // namespace hymap_detail

            template <class Fun, class Map>
            GT_TARGET GT_FORCE_INLINE constexpr auto transform(Fun && fun, Map && map) {
                return tuple_util::GT_TARGET_NAMESPACE_NAME::transform_index(
                    hymap_detail::adapter_f<Fun, get_keys<std::decay_t<Map>>>{std::forward<Fun>(fun)},
                    std::forward<Map>(map));
            }

            template <class Fun, class Map>
            GT_TARGET GT_FORCE_INLINE constexpr void for_each(Fun && fun, Map && map) {
                tuple_util::GT_TARGET_NAMESPACE_NAME::for_each_index(
                    hymap_detail::adapter_f<Fun, get_keys<std::decay_t<Map>>>{std::forward<Fun>(fun)},
                    std::forward<Map>(map));
            }

            // Concatenate several maps into one. The type of the result is infered from the type of the first map.
            // Precondition: keys should not overlap.
            template <class... Maps>
            GT_TARGET GT_FORCE_INLINE constexpr auto concat(Maps... maps) {
                static_assert(meta::is_set_fast<meta::concat<get_keys<Maps>...>>::value, GT_INTERNAL_ERROR);
                return tuple_util::concat_ex<impl_::concat_result_maker_f>(std::move(maps)...);
            }

            GT_TARGET GT_FORCE_INLINE constexpr keys<>::values<> concat() { return {}; }

            template <template <class...> class KeyCtor,
                class Keys,
                class Tup,
                class HyMapKeys = meta::rename<KeyCtor, Keys>>
            GT_TARGET GT_FORCE_INLINE constexpr auto convert_to(Tup && tup) {
                return tuple_util::convert_to<HyMapKeys::template values>(std::forward<Tup>(tup));
            }

            template <class Key, class Map>
            GT_TARGET GT_FORCE_INLINE constexpr auto canonicalize_and_remove_key(Map && map) {
                using res_t = from_meta_map<meta::mp_remove<hymap::to_meta_map<Map>, Key>>;
                using generators_t = meta::transform<hymap_detail::at_generator_f, get_keys<res_t>>;
                return tuple_util::generate<generators_t, res_t>(std::forward<Map>(map));
            }

            // This class holds two maps and models hymap content.
            // at_key returns the item of the primary map if the key is found there
            // otherwise it falls back to the secondary map.
            template <class Primary, class Secondary>
            class merged : tuple<Primary, Secondary> {
                using base_t = tuple<Primary, Secondary>;

                using meta_primary_t = to_meta_map<Primary>;
                using meta_secondary_t = to_meta_map<Secondary>;

                using primary_keys_t = meta::transform<meta::first, meta_primary_t>;
                using secondary_keys_t = meta::transform<meta::first, meta_secondary_t>;

                template <class Key,
                    class PrimaryItem = meta::mp_find<meta_primary_t, Key>,
                    class SecondaryItem = meta::mp_find<meta_secondary_t, Key>>
                using get_value_type = meta::second<meta::if_<std::is_void<PrimaryItem>, SecondaryItem, PrimaryItem>>;

                using keys_t = meta::dedup<meta::concat<primary_keys_t, secondary_keys_t>>;
                using values_t = meta::transform<get_value_type, keys_t>;

                template <size_t I, class Key = meta::at_c<keys_t, I>>
                using is_primary_index = meta::st_contains<primary_keys_t, Key>;

                template <size_t I, class Key = meta::at_c<keys_t, I>>
                using inner_index = meta::if_<is_primary_index<I>,
                    meta::st_position<primary_keys_t, Key>,
                    meta::st_position<secondary_keys_t, Key>>;

                template <size_t I,
                    class Key = meta::at_c<keys_t, I>,
                    class PrimaryPos = meta::st_position<primary_keys_t, Key>,
                    class SecondaryPos = meta::st_position<secondary_keys_t, Key>>
                using split_index = meta::if_c<(PrimaryPos::value < meta::length<primary_keys_t>::value),
                    meta::list<std::integral_constant<size_t, 0>, PrimaryPos>,
                    meta::list<std::integral_constant<size_t, 1>, SecondaryPos>>;

                friend values_t tuple_to_types(merged const &) { return {}; }

                friend keys_t hymap_get_keys(merged const &) { return {}; }

                friend struct merged_getter;

                GT_TARGET GT_FORCE_INLINE constexpr base_t const &base() const { return *this; }
                GT_TARGET GT_FORCE_INLINE constexpr base_t &base() { return *this; }

                template <size_t I>
                GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) source() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get < is_primary_index<I>::value ? 0 : 1 > (base());
                }
                template <size_t I>
                GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) source() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get < is_primary_index<I>::value ? 0 : 1 > (base());
                }

                template <size_t I>
                GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) get() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<inner_index<I>::value>(source<I>());
                }
                template <size_t I>
                GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) get() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<inner_index<I>::value>(source<I>());
                }

              public:
                merged() = default;

                GT_TARGET GT_FORCE_INLINE constexpr merged(Primary primary, Secondary secondary)
                    : base_t(std::move(primary), std::move(secondary)) {}

                GT_TARGET GT_FORCE_INLINE constexpr Primary const &primary() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<0>(base());
                }
                GT_TARGET GT_FORCE_INLINE constexpr Primary &primary() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<0>(base());
                }

                GT_TARGET GT_FORCE_INLINE constexpr Secondary const &secondary() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<1>(base());
                }
                GT_TARGET GT_FORCE_INLINE constexpr Secondary &secondary() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<1>(base());
                }

                // TODO:
                // - element wise ctor
                // - tuple_from_types
                // - hymap_from_keys_values
            };

            struct merged_getter {
                template <size_t I, class Primary, class Secondary>
                static GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) get(merged<Primary, Secondary> const &obj) {
                    return obj.template get<I>();
                }
                template <size_t I, class Primary, class Secondary>
                static GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) get(merged<Primary, Secondary> &obj) {
                    return obj.template get<I>();
                }
            };

            template <class Primary, class Secondary>
            merged_getter tuple_getter(merged<Primary, Secondary> const &);

            // merge the maps
            // unlike concat keys could overlap
            // in the case of overlap the value is taken from the primary map
            template <class Primary, class Secondary>
            GT_TARGET GT_FORCE_INLINE constexpr merged<Primary, Secondary> merge(Primary primary, Secondary secondary) {
                return {std::move(primary), std::move(secondary)};
            }

            template <class Primary, class... Secondaries>
            GT_TARGET GT_FORCE_INLINE constexpr auto merge(Primary primary, Secondaries... secondaries) {
                return merge(std::move(primary), merge(std::move(secondaries)...));
            }
        }
    } // namespace hymap
} // namespace gridtools

#endif // GT_TARGET_ITERATING
