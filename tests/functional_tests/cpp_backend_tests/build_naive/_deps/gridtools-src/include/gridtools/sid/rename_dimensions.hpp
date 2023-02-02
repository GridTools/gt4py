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
#include <utility>

#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "concept.hpp"
#include "delegate.hpp"
#include "unknown_kind.hpp"

namespace gridtools {
    namespace sid {
        namespace rename_dimensions_impl_ {

            template <class KeyMap>
            struct get_new_key_f {
                template <class Key>
                using apply = meta::second<meta::mp_find<KeyMap, Key, meta::list<Key, Key>>>;
            };

            template <class KeyMap, class Map>
            auto remap(Map map) {
                return hymap::convert_to<hymap::keys,
                    meta::transform<get_new_key_f<KeyMap>::template apply, get_keys<Map>>>(std::move(map));
            }

            template <class KeyMap, class Sid>
            struct renamed_sid : delegate<Sid> {
                using delegate<Sid>::delegate;
            };

            template <class...>
            struct stride_kind_wrapper {};

            template <class KeyMap, class Sid, class Kind = decltype(sid_get_strides_kind(std::declval<Sid const &>()))>
            meta::if_<std::is_same<Kind, unknown_kind>, Kind, stride_kind_wrapper<KeyMap, Kind>> sid_get_strides_kind(
                renamed_sid<KeyMap, Sid> const &);

            template <class KeyMap, class Sid>
            decltype(remap<KeyMap>(sid_get_strides(std::declval<Sid const &>()))) sid_get_strides(
                renamed_sid<KeyMap, Sid> const &obj) {
                return remap<KeyMap>(sid_get_strides(obj.m_impl));
            }

            template <class KeyMap, class Sid>
            decltype(remap<KeyMap>(sid_get_lower_bounds(std::declval<Sid const &>()))) sid_get_lower_bounds(
                renamed_sid<KeyMap, Sid> const &obj) {
                return remap<KeyMap>(sid_get_lower_bounds(obj.m_impl));
            }

            template <class KeyMap, class Sid>
            decltype(remap<KeyMap>(sid_get_upper_bounds(std::declval<Sid const &>()))) sid_get_upper_bounds(
                renamed_sid<KeyMap, Sid> const &obj) {
                return remap<KeyMap>(sid_get_upper_bounds(obj.m_impl));
            }

            template <class KeyMap, class Arr, std::enable_if_t<std::is_array_v<Arr>, int> = 0>
            auto sid_get_strides(renamed_sid<KeyMap, Arr &> const &obj) {
                return remap<KeyMap>(get_strides(obj.m_impl));
            }

            template <class KeyMap, class Arr, std::enable_if_t<std::is_array_v<Arr>, int> = 0>
            auto sid_get_lower_bounds(renamed_sid<KeyMap, Arr &> const &obj) {
                return remap<KeyMap>(get_lower_bounds(obj.m_impl));
            }

            template <class KeyMap, class Arr, std::enable_if_t<std::is_array_v<Arr>, int> = 0>
            auto sid_get_upper_bounds(renamed_sid<KeyMap, Arr &> const &obj) {
                return remap<KeyMap>(get_upper_bounds(obj.m_impl));
            }

            template <class...>
            struct make_key_map;

            template <>
            struct make_key_map<> : meta::list<> {};

            template <class Old, class New, class... Keys>
            struct make_key_map<Old, New, Keys...> {
                using type = meta::push_front<typename make_key_map<Keys...>::type, meta::list<Old, New>>;
            };

            // Keys parameters represent old and new dimension pairs
            // The order is the following old_key0, new_key0, old_key1, new_key1, ...
            template <class... Keys>
            struct rename_dimensions_f {
                template <class Sid>
                renamed_sid<typename make_key_map<Keys...>::type, Sid> operator()(Sid &&sid) const {
                    return {std::forward<Sid>(sid)};
                }
            };

            template <class... Keys>
            constexpr rename_dimensions_f<Keys...> rename_dimensions = {};

            template <class Keys, class Is = std::make_integer_sequence<int_t, meta::length<Keys>::value>>
            struct make_numbered_map;

            template <class... Keys, int_t... Is>
            struct make_numbered_map<meta::list<Keys...>, std::integer_sequence<int_t, Is...>> {
                using type = meta::list<meta::list<integral_constant<int_t, Is>, Keys>...>;
            };

            // rename_numbered_dimensions<a, b, c>(sid) is the same as
            // rename_dimensions<decltype(0_c), a, decltype(1_c), b decltype(2_c), c>(sid);
            template <class... Keys>
            struct rename_numbered_dimensions_f {
                template <class Sid>
                renamed_sid<typename make_numbered_map<meta::list<Keys...>>::type, Sid> operator()(Sid &&sid) const {
                    return {std::forward<Sid>(sid)};
                }
            };

            template <class... Keys>
            constexpr rename_numbered_dimensions_f<Keys...> rename_numbered_dimensions = {};
        } // namespace rename_dimensions_impl_
        using rename_dimensions_impl_::rename_dimensions;
        using rename_dimensions_impl_::rename_numbered_dimensions;
    } // namespace sid
} // namespace gridtools
