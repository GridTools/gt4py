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

#include <cassert>
#include <type_traits>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "data_store.hpp"

namespace gridtools {
    namespace storage {
        namespace storage_sid_impl_ {
            struct empty_ptr_diff {
                template <class T>
                friend constexpr GT_FUNCTION T *operator+(T *lhs, empty_ptr_diff) {
                    return lhs;
                }
            };

            template <class T>
            struct ptr_holder {
                T *m_val;
                GT_FUNCTION constexpr T *operator()() const { return m_val; }

                friend GT_FORCE_INLINE constexpr ptr_holder operator+(ptr_holder obj, int_t arg) {
                    return {obj.m_val + arg};
                }

                friend GT_FORCE_INLINE constexpr ptr_holder operator+(ptr_holder obj, empty_ptr_diff) { return obj; }
            };

            template <class Dim>
            struct bound_generator_f {
                template <class Lengths>
                auto operator()(Lengths const &lengths) const {
                    return tuple_util::get<Dim::value>(lengths);
                }
            };

            template <int... Is, class Bounds>
            auto filter_unmasked_bounds(layout_map<Is...>, Bounds const &bounds) {
                using all_keys_t = get_keys<Bounds>;
                using all_values_t = tuple_util::traits::to_types<Bounds>;
                using is_unmasked_t = meta::list<std::bool_constant<Is >= 0>...>;
                using all_items_t = meta::zip<is_unmasked_t, all_keys_t, all_values_t>;
                using items_t = meta::filter<meta::first, all_items_t>;
                using keys_t = meta::transform<meta::second, items_t>;
                using values_t = meta::transform<meta::third, items_t>;
                using res_t = hymap::from_keys_values<keys_t, values_t>;
                using generators_t = meta::transform<bound_generator_f, keys_t>;
                return tuple_util::generate<generators_t, res_t>(bounds);
            }
        } // namespace storage_sid_impl_

        /**
         *   The functions below make `data_store` model the `SID` concept
         */
        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        storage_sid_impl_::ptr_holder<typename DataStore::data_t> sid_get_origin(std::shared_ptr<DataStore> const &ds) {
            return {ds->get_target_ptr()};
        }

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        auto sid_get_strides(std::shared_ptr<DataStore> const &ds) {
            return ds->native_strides();
        }

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        typename DataStore::kind_t sid_get_strides_kind(std::shared_ptr<DataStore> const &);

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        meta::if_c<DataStore::layout_t::unmasked_length == 0, storage_sid_impl_::empty_ptr_diff, int_t>
        sid_get_ptr_diff(std::shared_ptr<DataStore> const &);

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        auto sid_get_lower_bounds(std::shared_ptr<DataStore> const &) {
            using layout_t = typename DataStore::layout_t;
            using bounds_t = meta::repeat_c<DataStore::ndims, tuple<integral_constant<int_t, 0>>>;
            return storage_sid_impl_::filter_unmasked_bounds(layout_t(), bounds_t());
        }

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        auto sid_get_upper_bounds(std::shared_ptr<DataStore> const &ds) {
            using layout_t = typename DataStore::layout_t;
            return storage_sid_impl_::filter_unmasked_bounds(layout_t(), ds->native_lengths());
        }
    } // namespace storage
} // namespace gridtools
