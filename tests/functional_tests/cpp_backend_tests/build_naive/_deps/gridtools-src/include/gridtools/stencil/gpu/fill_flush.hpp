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
#include <utility>

#include "../../common/defs.hpp"
#include "../../common/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../meta.hpp"
#include "../../sid/concept.hpp"
#include "../be_api.hpp"
#include "../common/caches.hpp"
#include "../common/dim.hpp"
#include "../global_parameter.hpp"
#include "../positional.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            namespace fill_flush {
                namespace impl_ {
                    template <class Cells>
                    using plh_map_from_cells =
                        meta::rename<be_api::merge_plh_maps, meta::transform<be_api::get_plh_map, Cells>>;

                    template <class Policy>
                    struct has_policy_f {
                        template <class PlhInfo>
                        using apply = meta::st_contains<typename PlhInfo::cache_io_policies_t, Policy>;
                    };

                    template <class Policy>
                    struct replace_policy_f {
                        template <class PlhInfo>
                        using apply = be_api::plh_info<typename PlhInfo::key_t,
                            typename PlhInfo::is_tmp_t,
                            typename PlhInfo::data_t,
                            typename PlhInfo::num_colors_t,
                            typename PlhInfo::is_const_t,
                            typename PlhInfo::extent_t,
                            meta::list<Policy>>;
                    };

                    template <class Policy, class PlhMap>
                    using filter_policy = meta::transform<replace_policy_f<Policy>::template apply,
                        meta::filter<has_policy_f<Policy>::template apply, PlhMap>>;

                    struct k_pos_key {};

                    enum class range { all, minus, plus };
                    enum class check { none, lo, hi };

                    template <class Ptrs>
                    GT_FUNCTION int_t get_k_pos(Ptrs const &ptrs) {
                        return *host_device::at_key<meta::list<k_pos_key>>(ptrs);
                    }

                    template <class PlhInfo, class Ptr, class Strides, class Offset>
                    GT_FUNCTION void shift_orig(Ptr &ptr, Strides const &strides, Offset offset) {
                        sid::shift(
                            ptr, sid::get_stride_element<meta::list<typename PlhInfo::plh_t>, dim::k>(strides), offset);
                    }

                    template <class PlhInfo, class Ptr, class Strides, class Offset>
                    GT_FUNCTION void shift_cached(Ptr &ptr, Strides const &strides, Offset offset) {
                        sid::shift(ptr, sid::get_stride_element<typename PlhInfo::key_t, dim::k>(strides), offset);
                    }

                    template <class PlhInfo, class Ptrs>
                    GT_FUNCTION auto get_orig(Ptrs const &ptrs) {
                        return host_device::at_key<meta::list<typename PlhInfo::plh_t>>(ptrs);
                    }

                    template <class PlhInfo, class Ptrs>
                    GT_FUNCTION auto get_cached(Ptrs const &ptrs) {
                        return host_device::at_key<typename PlhInfo::key_t>(ptrs);
                    }

                    template <class PlhInfo,
                        class Cached,
                        class Orig,
                        std::enable_if_t<
                            std::is_same_v<typename PlhInfo::cache_io_policies_t, meta::list<cache_io_policy::fill>>,
                            int> = 0>
                    GT_FUNCTION void sync(Cached cached, Orig orig) {
                        *cached = *orig;
                    }

                    template <class PlhInfo,
                        class Cached,
                        class Orig,
                        std::enable_if_t<
                            std::is_same_v<typename PlhInfo::cache_io_policies_t, meta::list<cache_io_policy::flush>>,
                            int> = 0>
                    GT_FUNCTION void sync(Cached cached, Orig orig) {
                        *orig = *cached;
                    }

                    template <class Plh, check>
                    struct bound {};

                    GT_FUNCTION bool is_k_valid(integral_constant<check, check::lo>, int_t k, int_t lim) {
                        return k >= lim;
                    }

                    GT_FUNCTION bool is_k_valid(integral_constant<check, check::hi>, int_t k, int_t lim) {
                        return k < lim;
                    }

                    template <class PlhInfo, range Range, check Check>
                    struct sync_fun {
                        using pos_key_t = meta::list<k_pos_key>;
                        using bound_key_t = meta::list<bound<typename PlhInfo::plh_t, Check>>;

                        template <class Deref = void, class Ptrs, class Strides>
                        GT_FUNCTION void operator()(Ptrs const &ptrs, Strides const &strides) {
                            using namespace literals;
                            auto orig = get_orig<PlhInfo>(ptrs);
                            auto cached = get_cached<PlhInfo>(ptrs);
                            auto lim = *host_device::at_key<bound_key_t>(ptrs);

                            using from_t = meta::if_c<Range == range::plus,
                                typename PlhInfo::extent_t::kplus,
                                typename PlhInfo::extent_t::kminus>;

                            shift_orig<PlhInfo>(orig, strides, from_t());
                            shift_cached<PlhInfo>(cached, strides, from_t());
                            int_t k = *host_device::at_key<pos_key_t>(ptrs) + from_t::value;

                            static constexpr int_t size = Range == range::all ? PlhInfo::extent_t::kplus::value -
                                                                                    PlhInfo::extent_t::kminus::value + 1
                                                                              : 1;
#pragma unroll
                            for (int_t i = 0; i < size; ++i) {
                                if (is_k_valid(integral_constant<check, Check>(), k, lim))
                                    sync<PlhInfo>(cached, orig);
                                shift_orig<PlhInfo>(orig, strides, 1_c);
                                shift_cached<PlhInfo>(cached, strides, 1_c);
                                ++k;
                            }
                        }

                        using plh_map_t = tuple<PlhInfo,
                            be_api::remove_caches_from_plh_info<PlhInfo>,
                            be_api::plh_info<pos_key_t,
                                std::false_type,
                                int_t const,
                                integral_constant<int_t, 0>,
                                std::true_type,
                                extent<>,
                                meta::list<>>,
                            be_api::plh_info<bound_key_t,
                                std::false_type,
                                int_t const,
                                integral_constant<int_t, 0>,
                                std::true_type,
                                extent<>,
                                meta::list<>>>;
                    };

                    template <class PlhInfo, range Range>
                    struct sync_fun<PlhInfo, Range, check::none> {
                        template <class Deref = void, class Ptrs, class Strides>
                        GT_FUNCTION void operator()(Ptrs const &ptrs, Strides const &strides) {
                            auto orig = get_orig<PlhInfo>(ptrs);
                            auto cached = get_cached<PlhInfo>(ptrs);
                            using offset_t = meta::if_c<Range == range::minus,
                                typename PlhInfo::extent_t::kminus,
                                typename PlhInfo::extent_t::kplus>;
                            shift_orig<PlhInfo>(orig, strides, offset_t());
                            shift_cached<PlhInfo>(cached, strides, offset_t());
                            sync<PlhInfo>(cached, orig);
                        }

                        using plh_map_t = tuple<PlhInfo, be_api::remove_caches_from_plh_info<PlhInfo>>;
                    };

                    template <class PlhInfo>
                    struct sync_fun<PlhInfo, range::all, check::none> {
                        template <class Deref = void, class Ptrs, class Strides>
                        GT_FUNCTION void operator()(Ptrs const &ptrs, Strides const &strides) {
                            using namespace literals;
                            auto orig = get_orig<PlhInfo>(ptrs);
                            auto cached = get_cached<PlhInfo>(ptrs);
                            using from_t = typename PlhInfo::extent_t::kminus;
                            static constexpr int_t size =
                                PlhInfo::extent_t::kplus::value - PlhInfo::extent_t::kminus::value + 1;
                            shift_orig<PlhInfo>(orig, strides, from_t());
                            shift_cached<PlhInfo>(cached, strides, from_t());
#pragma unroll
                            for (int_t i = 0; i < size; ++i) {
                                sync<PlhInfo>(cached, orig);
                                shift_orig<PlhInfo>(orig, strides, 1_c);
                                shift_cached<PlhInfo>(cached, strides, 1_c);
                            }
                        }

                        using plh_map_t = tuple<PlhInfo, be_api::remove_caches_from_plh_info<PlhInfo>>;
                    };

                    template <class FromLevel, class ToLevel, int_t Lim>
                    struct levels_are_close : std::false_type {};

                    constexpr int_t real_offset(int_t x) { return x > 0 ? x - 1 : x; }

                    template <uint_t Splitter, int_t OffsetLimit, int_t FromOffset, int_t ToOffset, int_t Lim>
                    struct levels_are_close<be_api::level<Splitter, FromOffset, OffsetLimit>,
                        be_api::level<Splitter, ToOffset, OffsetLimit>,
                        Lim> : std::bool_constant<(real_offset(ToOffset) - real_offset(FromOffset) < Lim)> {};

                    template <class PlhInfo,
                        class Execution,
                        class FirstInterval,
                        class LastInterval,
                        class CurInterval>
                    struct make_sync_fun {
                        static constexpr bool is_fill =
                            std::is_same_v<typename PlhInfo::cache_io_policies_t, meta::list<cache_io_policy::fill>>;
                        static constexpr bool is_first = std::is_same_v<FirstInterval, CurInterval>;
                        static constexpr bool is_last = std::is_same_v<LastInterval, CurInterval>;
                        static constexpr int_t minus = PlhInfo::extent_t::kminus::value;
                        static constexpr int_t plus = PlhInfo::extent_t::kplus::value;
                        static constexpr bool close_to_first =
                            levels_are_close<meta::first<FirstInterval>, meta::second<CurInterval>, -minus>::value;
                        static constexpr bool close_to_last =
                            levels_are_close<meta::first<CurInterval>, meta::second<LastInterval>, plus>::value;

                        //  Those static asserts are commented on purpose.
                        //  They trigger when the filling or the flushing of the k-cache could cause access violation in
                        //   the "inner" (runtime size) intervals due to the small offset limit.
                        //  We optimistically assume that the user knows what he is doing in this case.
                        //
                        //  static_assert(
                        //      levels_are_close<meta::first<FirstInterval>, meta::first<CurInterval>, -minus>::value ==
                        //      close_to_first, "offset_limit too small");
                        //  static_assert(
                        //      levels_are_close<meta::second<CurInterval>, meta::second<LastInterval>, plus>::value ==
                        //      close_to_last, "offset_limit too small");

                        static constexpr bool is_forward = !be_api::is_backward<Execution>::value;

                        static constexpr bool sync_all = is_forward == is_fill ? is_first : is_last;

                        static_assert(!sync_all || std::is_same_v<meta::first<CurInterval>, meta::second<CurInterval>>,
                            "offset_limit too small");

                        static constexpr range range_v = minus == plus           ? range::minus
                                                         : sync_all              ? range::all
                                                         : is_forward == is_fill ? range::plus
                                                                                 : range::minus;

                        static constexpr check check_v = minus == plus || PlhInfo::is_tmp_t::value ? check::none
                                                         : close_to_first                          ? check::lo
                                                         : close_to_last                           ? check::hi
                                                                                                   : check::none;

                        using type = sync_fun<PlhInfo, range_v, check_v>;
                    };

                    template <class PlhInfo, class Execution, class FirstInterval, class LastInterval>
                    struct make_cell_f {
                        template <class Interval,
                            class Fun =
                                typename make_sync_fun<PlhInfo, Execution, FirstInterval, LastInterval, Interval>::type>
                        using apply = be_api::cell<meta::list<Fun>,
                            Interval,
                            typename Fun::plh_map_t,
                            to_horizontal_extent<typename PlhInfo::extent_t>,
                            Execution,
                            std::false_type>;
                    };

                    template <class Intervals, class Execution>
                    struct make_stage_f {
                        template <class PlhInfo>
                        using apply = meta::transform<
                            make_cell_f<PlhInfo, Execution, meta::first<Intervals>, meta::last<Intervals>>::
                                template apply,
                            Intervals>;
                    };

                    template <class...>
                    struct transform_matrix;

                    template <class Matrix>
                    struct transform_matrix<Matrix> {
                        static_assert(meta::length<Matrix>::value > 0, GT_INTERNAL_ERROR);

                        using plh_map_t =
                            meta::rename<be_api::merge_plh_maps, meta::transform<plh_map_from_cells, Matrix>>;

                        using fill_map_t = filter_policy<cache_io_policy::fill, plh_map_t>;
                        using flush_map_t = filter_policy<cache_io_policy::flush, plh_map_t>;

                        using trimmed_matrix_t = meta::transpose<be_api::trim_interval_rows<meta::transpose<Matrix>>>;

                        using first_stage_cells_t = meta::first<trimmed_matrix_t>;
                        static_assert(meta::length<first_stage_cells_t>::value > 0, GT_INTERNAL_ERROR);

                        using execution_t = typename meta::first<first_stage_cells_t>::execution_t;

                        using intervals_t = meta::transform<be_api::get_interval, first_stage_cells_t>;

                        using type = meta::concat<
                            meta::transform<make_stage_f<intervals_t, execution_t>::template apply, fill_map_t>,
                            trimmed_matrix_t,
                            meta::transform<make_stage_f<intervals_t, execution_t>::template apply, flush_map_t>>;
                    };

                    template <class Matrices>
                    using transform_spec = meta::transform<meta::force<transform_matrix>::apply, Matrices>;

                    template <class Plh, class DataStores>
                    auto make_data_store(bound<Plh, check::lo>, DataStores const &data_stores) {
                        return global_parameter(
                            sid::get_lower_bound<dim::k>(sid::get_lower_bounds(at_key<Plh>(data_stores))));
                    }

                    template <class Plh, class DataStores>
                    auto make_data_store(bound<Plh, check::hi>, DataStores const &data_stores) {
                        return global_parameter(
                            sid::get_upper_bound<dim::k>(sid::get_upper_bounds(at_key<Plh>(data_stores))));
                    }

                    template <class DataStores>
                    positional<dim::k> make_data_store(k_pos_key, DataStores &&) {
                        return 0;
                    }

                    template <class DataStore>
                    struct is_missing_f {
                        template <class Plh>
                        using apply = std::negation<has_key<DataStore, Plh>>;
                    };

                    template <class PlhMap, class DataStores>
                    auto transform_data_stores(DataStores data_stores) {
                        using non_tmp_phs_t = meta::transform<be_api::get_plh,
                            meta::filter<meta::not_<be_api::get_is_tmp>::apply, PlhMap>>;
                        using plhs_t = meta::filter<is_missing_f<DataStores>::template apply, non_tmp_phs_t>;
                        auto extra = tuple_util::transform([&](auto plh) { return make_data_store(plh, data_stores); },
                            hymap::from_keys_values<plhs_t, plhs_t>());
                        return hymap::concat(std::move(data_stores), std::move(extra));
                    }

                    template <class Interval, int Lim = Interval::offset_limit>
                    using inner_interval = be_api::interval<be_api::level<meta::first<Interval>::splitter, Lim, Lim>,
                        be_api::level<meta::second<Interval>::splitter, -Lim, Lim>>;

                    template <class Spec, class Grid, class DataStores>
                    bool validate_k_bounds(Grid const &grid, DataStores const &data_stores) {
                        for_each<be_api::make_fused_view<Spec>>([&](auto mss) {
                            using mss_t = decltype(mss);
                            using interval_t = inner_interval<typename mss_t::interval_t>;
                            using is_backward_t = be_api::is_backward<typename mss_t::execution_t>;
                            using plh_map_t =
                                meta::filter<meta::not_<be_api::get_is_tmp>::apply, typename mss_t::plh_map_t>;
                            using fill_plhs_t = meta::filter<has_policy_f<cache_io_policy::fill>::apply, plh_map_t>;
                            using flush_plhs_t = meta::filter<has_policy_f<cache_io_policy::flush>::apply, plh_map_t>;
                            // Those asserts can trigger even if the user obeys the contract that the data is
                            // valid within computation area.
                            // Namely it can happen when k-cache windows are too big for the chosen offset limit.
                            for_each<meta::if_<is_backward_t, fill_plhs_t, flush_plhs_t>>(
                                [unchecked_area_begin = grid.k_start(interval_t()), &data_stores](auto info) {
                                    using plh_info_t = decltype(info);
                                    constexpr auto extent = plh_info_t::extent_t::kminus::value;
                                    auto lower_bound = sid::get_lower_bound<dim::k>(
                                        sid::get_lower_bounds(at_key<typename plh_info_t::plh_t>(data_stores)));
                                    assert(lower_bound <= unchecked_area_begin + extent);
                                });
                            for_each<meta::if_<is_backward_t, flush_plhs_t, fill_plhs_t>>(
                                [unchecked_area_end = grid.k_start(interval_t()) + grid.k_size(interval_t()),
                                    &data_stores](auto info) {
                                    using plh_info_t = decltype(info);
                                    constexpr auto extent = plh_info_t::extent_t::kplus::value;
                                    auto upper_bound = sid::get_upper_bound<dim::k>(
                                        sid::get_upper_bounds(at_key<typename plh_info_t::plh_t>(data_stores)));
                                    assert(upper_bound >= unchecked_area_end + extent);
                                });
                        });
                        return true;
                    } // namespace impl_

                } // namespace impl_
                using impl_::transform_data_stores;
                using impl_::transform_spec;
                using impl_::validate_k_bounds;
            } // namespace fill_flush
        }     // namespace gpu_backend
    }         // namespace stencil
} // namespace gridtools
