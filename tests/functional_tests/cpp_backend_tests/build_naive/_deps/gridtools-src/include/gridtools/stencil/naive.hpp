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

#include <memory>

#include "../common/defs.hpp"
#include "../common/for_each.hpp"
#include "../common/hymap.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/allocator.hpp"
#include "../sid/as_const.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/loop.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "be_api.hpp"
#include "common/dim.hpp"

namespace gridtools {
    namespace stencil {
        struct naive {
            template <class Spec, class Grid, class DataStores>
            friend void gridtools_backend_entry_point(naive, Spec, Grid const &grid, DataStores external_data_stores) {
                auto alloc = sid::host_device::allocator(&std::make_unique<char[]>);
                using stages_t = be_api::make_split_view<Spec>;
                using tmp_plh_map_t = be_api::remove_caches_from_plh_map<typename stages_t::tmp_plh_map_t>;
                auto temporaries = be_api::make_data_stores(tmp_plh_map_t(), [&](auto info) {
                    auto extent = info.extent();
                    auto interval = stages_t::interval();
                    auto num_colors = info.num_colors();
                    auto offsets = hymap::keys<dim::i, dim::j, dim::k>::make_values(-extent.minus(dim::i()),
                        -extent.minus(dim::j()),
                        -grid.k_start(interval) - extent.minus(dim::k()));
                    auto sizes = hymap::keys<dim::c, dim::k, dim::j, dim::i>::make_values(
                        num_colors, grid.k_size(interval, extent), grid.j_size(extent), grid.i_size(extent));
                    using stride_kind = meta::list<decltype(extent), decltype(num_colors)>;
                    return sid::shift_sid_origin(
                        sid::make_contiguous<decltype(info.data()), ptrdiff_t, stride_kind>(alloc, sizes), offsets);
                });
                auto data_stores = hymap::concat(external_data_stores, temporaries);
                using plh_map_t = typename stages_t::plh_map_t;
                using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;
                auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
                    [&](auto info) {
                        return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores));
                    },
                    plh_map_t()));
                auto origin = sid::get_origin(composite);
                auto strides = sid::get_strides(composite);
                for_each<stages_t>([&](auto stage) {
                    tuple_util::for_each(
                        [&](auto cell) {
                            auto ptr = origin();
                            auto extent = cell.extent();
                            auto interval = cell.interval();
                            sid::shift(ptr, sid::get_stride<dim::i>(strides), extent.minus(dim::i()));
                            sid::shift(ptr, sid::get_stride<dim::j>(strides), extent.minus(dim::j()));
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), grid.k_start(interval, cell.execution()));
                            auto i_loop = sid::make_loop<dim::i>(grid.i_size(extent));
                            auto j_loop = sid::make_loop<dim::j>(grid.j_size(extent));
                            auto k_loop = sid::make_loop<dim::k>(grid.k_size(interval), cell.k_step());
                            i_loop(j_loop(k_loop(cell)))(ptr, strides);
                        },
                        stage.cells());
                });
            }
        };
    } // namespace stencil
} // namespace gridtools
