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

#include "../common/for_each.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/concept.hpp"
#include "common/dim.hpp"
#include "common/extent.hpp"
#include "core/execution_types.hpp"
#include "core/interval.hpp"
#include "core/level.hpp"

namespace gridtools {
    namespace stencil {
        namespace be_api {
#define DEFINE_GETTER(property) \
    template <class T>          \
    using get_##property = typename T::property##_t

            DEFINE_GETTER(plh);
            DEFINE_GETTER(key);
            DEFINE_GETTER(is_tmp);
            DEFINE_GETTER(is_const);
            DEFINE_GETTER(data);
            DEFINE_GETTER(extent);
            DEFINE_GETTER(caches);
            DEFINE_GETTER(cache_io_policies);
            DEFINE_GETTER(num_colors);
            DEFINE_GETTER(funs);
            DEFINE_GETTER(interval);
            DEFINE_GETTER(plh_map);
            DEFINE_GETTER(execution);
            DEFINE_GETTER(need_sync);

#undef DEFINE_GETTER

            template <class Key,
                class IsTmp,
                class Data,
                class NumColors,
                class IsConst,
                class Extent,
                class CacheIoPolicies>
            struct plh_info;

            template <class Plh,
                class... Caches,
                class IsTmp,
                class Data,
                class NumColors,
                class IsConst,
                class Extent,
                class... CacheIoPolicies>
            struct plh_info<meta::list<Plh, Caches...>,
                IsTmp,
                Data,
                NumColors,
                IsConst,
                Extent,
                meta::list<CacheIoPolicies...>> {
                using key_t = meta::list<Plh, Caches...>;
                using plh_t = Plh;
                using caches_t = meta::list<Caches...>;
                using is_tmp_t = IsTmp;
                using data_t = Data;
                using num_colors_t = NumColors;
                using is_const_t = IsConst;
                using extent_t = Extent;
                using cache_io_policies_t = meta::list<CacheIoPolicies...>;

                static GT_FUNCTION key_t key() { return {}; }
                static GT_FUNCTION plh_t plh() { return {}; }
                static GT_FUNCTION caches_t caches() { return {}; }
                static GT_FUNCTION is_tmp_t is_tmp() { return {}; }
                static data_t data();
                static GT_FUNCTION num_colors_t num_colors() { return {}; }
                static GT_FUNCTION is_const_t is_const() { return {}; }
                static GT_FUNCTION extent_t extent() { return {}; }
                static GT_FUNCTION cache_io_policies_t cache_io_policies() { return {}; }
            };

            template <template <class...> class GetKey = get_plh, class PlhMap, class Fun>
            auto make_data_stores(PlhMap, Fun &&fun) {
                return tuple_util::transform(
                    std::forward<Fun>(fun), hymap::from_keys_values<meta::transform<GetKey, PlhMap>, PlhMap>());
            }

            template <class Items, class Grid>
            auto make_k_sizes(Items, const Grid &grid) {
                return tuple_util::transform([&](auto item) { return grid.k_size(item.interval()); }, Items());
            }

            namespace lazy {
                template <class...>
                struct merge_plh_infos;

                template <class Key,
                    class IsTmp,
                    class Data,
                    class NumColors,
                    class... IsConsts,
                    class... Extents,
                    class... CacheIoPolicyLists>
                struct merge_plh_infos<
                    plh_info<Key, IsTmp, Data, NumColors, IsConsts, Extents, CacheIoPolicyLists>...> {
                    using type = plh_info<Key,
                        IsTmp,
                        Data,
                        NumColors,
                        typename std::conjunction<IsConsts...>::type,
                        enclosing_extent<Extents...>,
                        meta::dedup<meta::concat<CacheIoPolicyLists...>>>;
                };

                template <class...>
                struct remove_caches_from_plh_info;

                template <class Plh,
                    class... Caches,
                    class IsTmp,
                    class Data,
                    class NumColor,
                    class IsConst,
                    class Extent,
                    class... CacheIoPolicies>
                struct remove_caches_from_plh_info<plh_info<meta::list<Plh, Caches...>,
                    IsTmp,
                    Data,
                    NumColor,
                    IsConst,
                    Extent,
                    meta::list<CacheIoPolicies...>>> {
                    using type = plh_info<meta::list<Plh>, IsTmp, Data, NumColor, IsConst, Extent, meta::list<>>;
                };
            } // namespace lazy
            GT_META_DELEGATE_TO_LAZY(merge_plh_infos, class... Ts, Ts...);
            GT_META_DELEGATE_TO_LAZY(remove_caches_from_plh_info, class... Ts, Ts...);

            template <class... Maps>
            using merge_plh_maps = meta::mp_make<merge_plh_infos, meta::concat<Maps...>>;

            template <class Map>
            using remove_caches_from_plh_map =
                meta::mp_make<merge_plh_infos, meta::transform<remove_caches_from_plh_info, Map>>;

            template <class Deref, class Ptr, class Strides>
            struct run_f {
                Ptr const &m_ptr;
                Strides const &m_strides;

                template <class Fun>
                GT_FUNCTION void operator()(Fun fun) const {
                    fun.template operator()<Deref>(m_ptr, m_strides);
                }
            };

            template <class Funs, class Interval, class PlhMap, class Extent, class Execution, class NeedSync>
            struct cell {
                using funs_t = Funs;
                using interval_t = Interval;
                using plh_map_t = PlhMap;
                using extent_t = Extent;
                using execution_t = Execution;
                using need_sync_t = NeedSync;

                using plhs_t = meta::transform<get_plh, plh_map_t>;
                using k_step_t = integral_constant<int_t, core::is_backward<Execution>::value ? -1 : 1>;

                static GT_FUNCTION Funs funs() { return {}; }
                static GT_FUNCTION Interval interval() { return {}; }
                static GT_FUNCTION PlhMap plh_map() { return {}; }
                static GT_FUNCTION Extent extent() { return {}; }
                static GT_FUNCTION Execution execution() { return {}; }
                static GT_FUNCTION NeedSync need_sync() { return {}; }

                static GT_FUNCTION plhs_t plhs() { return {}; }
                static GT_FUNCTION k_step_t k_step() { return {}; }

                template <class Deref = void, class Ptr, class Strides>
                GT_FUNCTION void operator()(Ptr const &ptr, Strides const &strides) const {
                    host_device::for_each<Funs>(run_f<Deref, Ptr, Strides>{ptr, strides});
                }

                template <class Ptr, class Strides>
                static GT_FUNCTION void inc_k(Ptr &ptr, Strides const &strides) {
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), k_step());
                }
            };

            template <class... Ts>
            struct can_fuse_intervals : std::false_type {};

            template <class Funs, class... Intervals, class PlhMap, class Extent, class Execution, class NeedSync>
            struct can_fuse_intervals<cell<Funs, Intervals, PlhMap, Extent, Execution, NeedSync>...> : std::true_type {
            };

            template <class...>
            struct can_fuse_stages : std::false_type {};

            template <class... NeedSyncs>
            struct can_fuse_need_syncs;

            template <class First, class... NeedSyncs>
            struct can_fuse_need_syncs<First, NeedSyncs...> : std::conjunction<std::negation<NeedSyncs>...> {};

            template <class... Funs, class Interval, class... PlhMaps, class Extent, class Execution, class... NeedSync>
            struct can_fuse_stages<cell<Funs, Interval, PlhMaps, Extent, Execution, NeedSync>...>
                : can_fuse_need_syncs<NeedSync...> {};

            namespace lazy {
                template <class...>
                struct fuse_intervals;

                template <class Funs, class... Intervals, class PlhMap, class Extent, class Execution, class NeedSync>
                struct fuse_intervals<cell<Funs, Intervals, PlhMap, Extent, Execution, NeedSync>...> {
                    using type = cell<Funs, core::concat_intervals<Intervals...>, PlhMap, Extent, Execution, NeedSync>;
                };

                template <class...>
                struct fuse_stages;

                template <class... Funs,
                    class Interval,
                    class... PlhMaps,
                    class Extent,
                    class Execution,
                    class... NeedSync>
                struct fuse_stages<cell<Funs, Interval, PlhMaps, Extent, Execution, NeedSync>...> {
                    using type = cell<meta::concat<Funs...>,
                        Interval,
                        merge_plh_maps<PlhMaps...>,
                        Extent,
                        Execution,
                        std::disjunction<NeedSync...>>;
                };
            } // namespace lazy
            GT_META_DELEGATE_TO_LAZY(fuse_intervals, class... Ts, Ts...);
            GT_META_DELEGATE_TO_LAZY(fuse_stages, class... Ts, Ts...);

            template <class Cell>
            using is_cell_empty = meta::is_empty<typename Cell::funs_t>;

            template <template <class...> class Pred>
            struct row_predicate_f {
                template <class... Rows>
                using apply = meta::all<meta::transform<Pred, Rows...>>;
            };

            template <template <class...> class F>
            struct row_fuse_f {
                template <class... Rows>
                using apply = meta::transform<F, Rows...>;
            };

            template <class Cells>
            using has_funs = std::negation<meta::all_of<is_cell_empty, Cells>>;

            template <template <class...> class Pred, template <class...> class F, class Matrix>
            using fuse_rows = meta::group<row_predicate_f<Pred>::template apply, row_fuse_f<F>::template apply, Matrix>;

            template <class TransposedMatrix>
            using fuse_interval_rows = fuse_rows<can_fuse_intervals, fuse_intervals, TransposedMatrix>;

            template <class TransposedMatrix>
            using trim_interval_rows = meta::trim<row_predicate_f<is_cell_empty>::apply, TransposedMatrix>;

            template <class TransposedMatrix>
            using compress_interval_rows = trim_interval_rows<fuse_interval_rows<TransposedMatrix>>;

            template <class Cells>
            using compress_intervals =
                meta::trim<is_cell_empty, meta::group<can_fuse_intervals, fuse_intervals, Cells>>;

            template <class Matrix>
            using fuse_stage_rows = fuse_rows<can_fuse_stages, fuse_stages, Matrix>;

            template <class... Ts>
            struct interval_info;

            template <class... Funs,
                class Interval,
                class... PlhMaps,
                class... Extent,
                class Execution,
                class... NeedSync>
            struct interval_info<cell<Funs, Interval, PlhMaps, Extent, Execution, NeedSync>...> {
                using interval_t = Interval;
                using plh_map_t = merge_plh_maps<PlhMaps...>;
                using extent_t = enclosing_extent<Extent...>;
                using execution_t = Execution;
                using plhs_t = meta::transform<get_plh, plh_map_t>;
                using keys_t = meta::transform<get_key, plh_map_t>;
                using k_step_t = typename meta::first<interval_info>::k_step_t;

                using cells_t = meta::rename<tuple, meta::filter<meta::not_<is_cell_empty>::apply, interval_info>>;

                static GT_FUNCTION execution_t execution() { return {}; }
                static GT_FUNCTION extent_t extent() { return {}; }
                static GT_FUNCTION plh_map_t plh_map() { return {}; }
                static GT_FUNCTION plhs_t plhs() { return {}; }
                static GT_FUNCTION interval_t interval() { return {}; }
                static GT_FUNCTION k_step_t k_step() { return {}; }
                static GT_FUNCTION cells_t cells() { return {}; }

                template <class Ptr, class Strides>
                static GT_FUNCTION void inc_k(Ptr &ptr, Strides const &strides) {
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), k_step());
                }
            };

            template <class... IntervalInfos>
            class fused_view_item {
                static_assert(sizeof...(IntervalInfos) > 0, GT_INTERNAL_ERROR);
                static_assert(std::conjunction<meta::is_instantiation_of<interval_info, IntervalInfos>...>::value,
                    GT_INTERNAL_ERROR);
                static_assert(meta::are_same<typename meta::length<IntervalInfos>::type...>::value, GT_INTERNAL_ERROR);

                using item_t = meta::first<meta::list<IntervalInfos...>>;

              public:
                using execution_t = typename item_t::execution_t;
                using extent_t = enclosing_extent<typename IntervalInfos::extent_t...>;
                using plh_map_t = merge_plh_maps<typename IntervalInfos::plh_map_t...>;
                using plhs_t = meta::transform<get_plh, plh_map_t>;
                using keys_t = meta::transform<get_key, plh_map_t>;
                using interval_t = core::concat_intervals<typename IntervalInfos::interval_t...>;
                using k_step_t = typename item_t::k_step_t;

                using interval_infos_t = meta::if_<core::is_backward<execution_t>,
                    meta::reverse<meta::rename<tuple, fused_view_item>>,
                    meta::rename<tuple, fused_view_item>>;

                static GT_FUNCTION execution_t execution() { return {}; }
                static GT_FUNCTION extent_t extent() { return {}; }
                static GT_FUNCTION plh_map_t plh_map() { return {}; }
                static GT_FUNCTION plhs_t plhs() { return {}; }
                static GT_FUNCTION interval_t interval() { return {}; }
                static GT_FUNCTION k_step_t k_step() { return {}; }

                static GT_FUNCTION interval_infos_t interval_infos() { return {}; }
            };

            template <class... Cells>
            class split_view_item {
                static_assert(sizeof...(Cells) > 0, GT_INTERNAL_ERROR);
                using cell_t = meta::first<split_view_item>;

              public:
                using execution_t = typename cell_t::execution_t;
                using extent_t = enclosing_extent<typename Cells::extent_t...>;
                using plh_map_t = merge_plh_maps<typename Cells::plh_map_t...>;
                using plhs_t = meta::transform<get_plh, plh_map_t>;
                using interval_t = core::concat_intervals<typename Cells::interval_t...>;
                using k_step_t = typename cell_t::k_step_t;

                using cells_t = meta::if_<core::is_backward<execution_t>,
                    meta::reverse<meta::rename<tuple, split_view_item>>,
                    meta::rename<tuple, split_view_item>>;

                static GT_FUNCTION execution_t execution() { return {}; }
                static GT_FUNCTION extent_t extent() { return {}; }
                static GT_FUNCTION plh_map_t plh_map() { return {}; }
                static GT_FUNCTION plhs_t plhs() { return {}; }
                static GT_FUNCTION interval_t interval() { return {}; }
                static GT_FUNCTION k_step_t k_step() { return {}; }

                static GT_FUNCTION cells_t cells() { return {}; }
            };

            template <class... Items>
            struct aggregated_view {
                using plh_map_t = merge_plh_maps<typename Items::plh_map_t...>;
                using plhs_t = meta::transform<get_plh, plh_map_t>;
                using keys_t = meta::transform<get_keys, plh_map_t>;

                using tmp_plh_map_t = meta::filter<get_is_tmp, plh_map_t>;
                using tmp_plhs_t = meta::transform<get_plh, tmp_plh_map_t>;

                using interval_t = core::enclosing_interval<typename Items::interval_t...>;

                static GT_FUNCTION tmp_plh_map_t tmp_plh_map() { return {}; }
                static GT_FUNCTION interval_t interval() { return {}; }
            };

            template <class Matrix>
            using make_fused_view_item = meta::rename<fused_view_item,
                meta::transform<meta::rename<interval_info>::apply,
                    compress_interval_rows<meta::transpose<fuse_stage_rows<Matrix>>>>>;

            template <class Matrices>
            using make_fused_view = meta::rename<aggregated_view, meta::transform<make_fused_view_item, Matrices>>;

            template <class Cells>
            using make_split_view_item = meta::rename<split_view_item, compress_intervals<Cells>>;

            template <class Matrices>
            using make_split_view = meta::rename<aggregated_view,
                meta::transform<make_split_view_item, meta::flatten<meta::transform<fuse_stage_rows, Matrices>>>>;

            using core::is_backward;
            using core::is_forward;
            using core::is_parallel;

            // used in gpu/fill_flush. TODO: get rid of that?
            using core::interval;
            using core::level;
        } // namespace be_api
    }     // namespace stencil
} // namespace gridtools
