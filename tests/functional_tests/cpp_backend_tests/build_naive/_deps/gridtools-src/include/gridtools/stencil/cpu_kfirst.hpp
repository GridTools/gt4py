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
#include <utility>

#include "../common/defs.hpp"
#include "../common/for_each.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/allocator.hpp"
#include "../sid/as_const.hpp"
#include "../sid/block.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/loop.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "../thread_pool/concept.hpp"
#include "../thread_pool/omp.hpp"
#include "be_api.hpp"
#include "common/dim.hpp"

namespace gridtools {
    namespace stencil {
        namespace cpu_kfirst_backend {
            template <class ThreadPool, class Stage, class Grid, class DataStores>
            auto make_stage_loop(ThreadPool, Stage, Grid const &grid, DataStores &data_stores) {
                using extent_t = typename Stage::extent_t;

                using plh_map_t = typename Stage::plh_map_t;
                using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;
                auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
                    [&](auto info) GT_FORCE_INLINE_LAMBDA {
                        return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores));
                    },
                    Stage::plh_map()));
                using ptr_diff_t = sid::ptr_diff_type<decltype(composite)>;

                auto strides = sid::get_strides(composite);
                ptr_diff_t offset{};
                sid::shift(offset, sid::get_stride<dim::i>(strides), extent_t::minus(dim::i()));
                sid::shift(offset, sid::get_stride<dim::j>(strides), extent_t::minus(dim::j()));
                sid::shift(
                    offset, sid::get_stride<dim::k>(strides), grid.k_start(Stage::interval(), Stage::execution()));

                auto shift_back = -grid.k_size(Stage::interval()) * Stage::k_step();
                auto k_sizes = tuple_util::transform(
                    [&](auto cell) GT_FORCE_INLINE_LAMBDA { return grid.k_size(cell.interval()); }, Stage::cells());
                auto k_loop = [k_sizes = std::move(k_sizes), shift_back](auto &ptr, auto const &strides)
                                  GT_FORCE_INLINE_LAMBDA {
                                      tuple_util::for_each(
                                          [&ptr, &strides](auto cell, auto size) GT_FORCE_INLINE_LAMBDA {
                                              for (int_t k = 0; k < size; ++k) {
                                                  cell(ptr, strides);
                                                  cell.inc_k(ptr, strides);
                                              }
                                          },
                                          Stage::cells(),
                                          k_sizes);
                                      sid::shift(ptr, sid::get_stride<dim::k>(strides), shift_back);
                                  };
                return [origin = sid::get_origin(composite) + offset,
                           strides = std::move(strides),
                           k_loop = std::move(k_loop)](int_t i_block, int_t j_block, int_t i_size, int_t j_size) {
                    ptr_diff_t offset{};
                    sid::shift(
                        offset, sid::get_stride<dim::thread>(strides), thread_pool::get_thread_num(ThreadPool()));
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), i_block);
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), j_block);
                    auto i_loop = sid::make_loop<dim::i>(extent_t::extend(dim::i(), i_size));
                    auto j_loop = sid::make_loop<dim::j>(extent_t::extend(dim::j(), j_size));
                    i_loop(j_loop(k_loop))(origin() + offset, strides);
                };
            }

            template <class IBlockSize = integral_constant<int_t, 8>,
                class JBlockSize = integral_constant<int_t, 8>,
                class ThreadPool = thread_pool::omp>
            struct cpu_kfirst {};

            template <class IBlockSize, class JBlockSize, class ThreadPool, class Spec, class Grid, class DataStores>
            void gridtools_backend_entry_point(cpu_kfirst<IBlockSize, JBlockSize, ThreadPool>,
                Spec,
                Grid const &grid,
                DataStores external_data_stores) {
                using stages_t = be_api::make_split_view<Spec>;

                auto alloc = sid::cached_allocator(&std::make_unique<char[]>);

                using tmp_plh_map_t = be_api::remove_caches_from_plh_map<typename stages_t::tmp_plh_map_t>;
                auto temporaries = be_api::make_data_stores(tmp_plh_map_t(), [&grid, &alloc](auto info) {
                    auto extent = info.extent();
                    auto interval = stages_t::interval();
                    auto num_colors = info.num_colors();
                    auto offsets = hymap::keys<dim::i, dim::j, dim::k>::make_values(-extent.minus(dim::i()),
                        -extent.minus(dim::j()),
                        -grid.k_start(interval) - extent.minus(dim::k()));
                    auto sizes = hymap::keys<dim::c, dim::k, dim::j, dim::i, dim::thread>::make_values(num_colors,
                        grid.k_size(interval, extent),
                        extent.extend(dim::j(), JBlockSize()),
                        extent.extend(dim::i(), IBlockSize()),
                        thread_pool::get_max_threads(ThreadPool()));

                    using stride_kind = meta::list<decltype(extent), decltype(num_colors)>;
                    return sid::shift_sid_origin(
                        sid::make_contiguous<decltype(info.data()), int_t, stride_kind>(alloc, sizes), offsets);
                });

                auto blocked_external_data_stores = tuple_util::transform(
                    [&](auto &&data_store) GT_FORCE_INLINE_LAMBDA {
                        return sid::block(std::forward<decltype(data_store)>(data_store),
                            hymap::keys<dim::i, dim::j>::values<IBlockSize, JBlockSize>());
                    },
                    std::move(external_data_stores));

                auto data_stores = hymap::concat(std::move(blocked_external_data_stores), std::move(temporaries));

                auto stage_loops = tuple_util::transform(
                    [&](auto stage)
                        GT_FORCE_INLINE_LAMBDA { return make_stage_loop(ThreadPool(), stage, grid, data_stores); },
                    meta::rename<tuple, stages_t>());

                int_t total_i = grid.i_size();
                int_t total_j = grid.j_size();

                int_t NBI = (total_i + IBlockSize::value - 1) / IBlockSize::value;
                int_t NBJ = (total_j + JBlockSize::value - 1) / JBlockSize::value;

                thread_pool::parallel_for_loop(
                    ThreadPool(),
                    [&](auto bj, auto bi) {
                        int_t i_size = bi + 1 == NBI ? total_i - bi * IBlockSize::value : IBlockSize::value;
                        int_t j_size = bj + 1 == NBJ ? total_j - bj * JBlockSize::value : JBlockSize::value;
                        tuple_util::for_each(
                            [=](auto &&fun) GT_FORCE_INLINE_LAMBDA { fun(bi, bj, i_size, j_size); }, stage_loops);
                    },
                    NBJ,
                    NBI);
            }
        } // namespace cpu_kfirst_backend
        using cpu_kfirst_backend::cpu_kfirst;
    } // namespace stencil
} // namespace gridtools
