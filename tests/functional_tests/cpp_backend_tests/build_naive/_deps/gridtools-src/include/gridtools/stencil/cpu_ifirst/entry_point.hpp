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

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../sid/as_const.hpp"
#include "../../sid/block.hpp"
#include "../../sid/composite.hpp"
#include "../../sid/concept.hpp"
#include "../../thread_pool/omp.hpp"
#include "../be_api.hpp"
#include "../common/dim.hpp"
#include "execinfo.hpp"
#include "loops.hpp"
#include "pos3.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace stencil {
        namespace cpu_ifirst_backend {
            template <class ThreadPool = thread_pool::omp>
            struct cpu_ifirst {
                template <class Spec, class Grid, class DataStores>
                friend void gridtools_backend_entry_point(
                    cpu_ifirst, Spec, Grid const &grid, DataStores external_data_stores) {
                    using stages_t = be_api::make_split_view<Spec>;
                    using all_parrallel_t = typename meta::all_of<be_api::is_parallel,
                        meta::transform<be_api::get_execution, stages_t>>::type;
                    using enclosing_extent_t = meta::rename<enclosing_extent,
                        meta::transform<be_api::get_extent, typename stages_t::plh_map_t>>;
                    using fuse_all_t =
                        std::bool_constant<all_parrallel_t::value && enclosing_extent_t::kminus::value == 0 &&
                                           enclosing_extent_t::kplus::value == 0>;

                    tmp_allocator alloc;

                    execinfo info(ThreadPool(), grid);

                    using tmp_plh_map_t = be_api::remove_caches_from_plh_map<typename stages_t::tmp_plh_map_t>;
                    auto temporaries = be_api::make_data_stores(tmp_plh_map_t(),
                        [&alloc,
                            block_size = make_pos3(
                                (size_t)info.i_block_size(), (size_t)info.j_block_size(), (size_t)grid.k_size())](
                            auto info) {
                            return make_tmp_storage<decltype(info.data()),
                                decltype(info.extent()),
                                fuse_all_t::value,
                                ThreadPool>(alloc, block_size);
                        });

                    auto blocked_externals = tuple_util::transform(
                        [block_size = hymap::keys<dim::i, dim::j>::make_values(
                             info.i_block_size(), info.j_block_size())](auto &&data_store) {
                            return sid::block(std::forward<decltype(data_store)>(data_store), block_size);
                        },
                        std::move(external_data_stores));

                    auto data_stores = hymap::concat(std::move(blocked_externals), std::move(temporaries));

                    auto loops = tuple_util::transform(
                        [&](auto stage) {
                            using stage_t = decltype(stage);
                            auto k_sizes = tuple_util::transform(
                                [&](auto cell) { return grid.k_size(cell.interval()); }, stage_t::cells());

                            using plh_map_t = typename stage_t::plh_map_t;
                            using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;
                            auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
                                [&](auto info) {
                                    return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores));
                                },
                                stage_t::plh_map()));
                            return make_loop<ThreadPool, stage_t>(
                                fuse_all_t(), grid, std::move(composite), std::move(k_sizes));
                        },
                        meta::rename<tuple, stages_t>());

                    run_loops<ThreadPool>(fuse_all_t(), grid, std::move(loops));
                }
            };
        } // namespace cpu_ifirst_backend
        using cpu_ifirst_backend::cpu_ifirst;
    } // namespace stencil
} // namespace gridtools
