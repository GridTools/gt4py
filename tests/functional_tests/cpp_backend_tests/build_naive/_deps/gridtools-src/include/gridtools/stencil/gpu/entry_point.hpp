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

#include "../../common/cuda_type_traits.hpp"
#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../sid/allocator.hpp"
#include "../../sid/as_const.hpp"
#include "../../sid/block.hpp"
#include "../../sid/composite.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/sid_shift_origin.hpp"
#include "../be_api.hpp"
#include "../common/caches.hpp"
#include "../common/dim.hpp"
#include "../common/extent.hpp"
#include "fill_flush.hpp"
#include "ij_cache.hpp"
#include "k_cache.hpp"
#include "launch_kernel.hpp"
#include "make_kernel_fun.hpp"
#include "shared_allocator.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            template <class PlhInfo>
            using is_not_cached = meta::is_empty<typename PlhInfo::caches_t>;

            template <class Is, class... Funs>
            struct multi_kernel;

            template <class... Funs, size_t... Is>
            struct multi_kernel<std::index_sequence<Is...>, Funs...> {
                tuple<Funs...> m_funs;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator const &validator) const {
                    (void)(int[]){(tuple_util::device::get<Is>(m_funs)(i_block, j_block, validator), 0)...};
                }
            };

            template <class Fun>
            Fun make_multi_kernel(tuple<Fun> tup) {
                return tuple_util::get<0>(std::move(tup));
            }

            template <class... Funs>
            multi_kernel<std::index_sequence_for<Funs...>, Funs...> make_multi_kernel(tuple<Funs...> tup) {
                return {std::move(tup)};
            }

            template <int_t BlockSize>
            std::enable_if_t<BlockSize == 0, int_t> blocks_required_z(int_t) {
                return 1;
            }

            template <int_t BlockSize>
            std::enable_if_t<BlockSize != 0, int_t> blocks_required_z(int_t nz) {
                return (nz + BlockSize - 1) / BlockSize;
            }

            struct dummy_info {
                using extent_t = extent<>;
            };

            template <class PlhMap>
            struct get_extent_f {
                template <class Key>
                using apply = typename meta::mp_find<PlhMap, Key, dummy_info>::extent_t;
            };

            template <class MaxExtent, class PlhMap, int_t BlockSize, bool IsParallel, class... Funs>
            struct kernel {
                tuple<Funs...> m_funs;
                size_t m_shared_memory_size;

                template <class Backend, class Grid, class Kernel>
                Kernel launch_or_fuse(Backend, Grid const &grid, Kernel kernel) && {
                    launch_kernel<MaxExtent, Backend::i_block_size_t::value, Backend::j_block_size_t::value>(
                        grid.i_size(),
                        grid.j_size(),
                        blocks_required_z<BlockSize>(grid.k_size()),
                        make_multi_kernel(std::move(m_funs)),
                        m_shared_memory_size);
                    return kernel;
                }

                template <class OtherPlhMap,
                    class Backend,
                    class Grid,
                    class Fun,
                    class OutKeys =
                        meta::transform<be_api::get_key, meta::filter<meta::not_<be_api::get_is_const>::apply, PlhMap>>,
                    class Extents = meta::transform<get_extent_f<OtherPlhMap>::template apply, OutKeys>,
                    class Extent = meta::rename<enclosing_extent, Extents>,
                    std::enable_if_t<Extent::iminus::value == 0 && Extent::iplus::value == 0 &&
                                         Extent::jminus::value == 0 && Extent::jplus::value == 0 &&
                                         (!IsParallel || (Extent::kminus::value == 0 && Extent::kplus::value == 0)),
                        int> = 0>
                kernel<MaxExtent, be_api::merge_plh_maps<PlhMap, OtherPlhMap>, BlockSize, IsParallel, Funs..., Fun>
                launch_or_fuse(
                    Backend, Grid const &grid, kernel<MaxExtent, OtherPlhMap, BlockSize, IsParallel, Fun> kernel) && {
                    return {tuple_util::push_back(std::move(m_funs), tuple_util::get<0>(std::move(kernel.m_funs))),
                        std::max(m_shared_memory_size, kernel.m_shared_memory_size)};
                }
            };

            struct no_kernel {
                template <class Backend, class Grid, class Kernel>
                Kernel launch_or_fuse(Backend, Grid &&, Kernel kernel) && {
                    return kernel;
                }
            };

            template <class Keys>
            struct deref_f {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
                template <class Key, class T>
                GT_FUNCTION std::enable_if_t<is_texture_type<T>::value && meta::st_contains<Keys, Key>::value, T>
                operator()(Key, T const *ptr) const {
                    return __ldg(ptr);
                }
#endif
                template <class Key, class Ptr>
                GT_FUNCTION decltype(auto) operator()(Key, Ptr ptr) const {
                    return *ptr;
                }
            };

            template <class IBlockSize = integral_constant<int_t, 64>,
                class JBlockSize = integral_constant<int_t, 8>,
                class KBlockSize = integral_constant<int_t, 1>>
            struct gpu {
                using i_block_size_t = IBlockSize;
                using j_block_size_t = JBlockSize;

                template <class Msses, class Grid, class Allocator>
                static auto make_temporaries(Grid const &grid, Allocator &allocator) {
                    using plh_map_t = meta::filter<is_not_cached, typename Msses::tmp_plh_map_t>;
                    using extent_t = meta::rename<enclosing_extent, meta::transform<be_api::get_extent, plh_map_t>>;
                    return tuple_util::transform(
                        [&allocator,
                            n_blocks_i = (grid.i_size() + IBlockSize() - 1) / IBlockSize(),
                            n_blocks_j = (grid.j_size() + JBlockSize() - 1) / JBlockSize(),
                            k_size = grid.k_size()](auto info) {
                            using info_t = decltype(info);
                            return make_tmp_storage<typename info_t::data_t>(typename info_t::num_colors_t(),
                                IBlockSize(),
                                JBlockSize(),
                                extent_t(),
                                n_blocks_i,
                                n_blocks_j,
                                k_size,
                                allocator);
                        },
                        hymap::from_keys_values<meta::transform<be_api::get_plh, plh_map_t>, plh_map_t>());
                }

                template <class DataStoreMap>
                static auto block(DataStoreMap data_stores) {
                    return tuple_util::transform(
                        [=](auto &&src) {
                            return sid::block(std::forward<decltype(src)>(src),
                                hymap::keys<dim::i, dim::j>::make_values(IBlockSize(), JBlockSize()));
                        },
                        std::move(data_stores));
                }

                template <class Deref, class Mss, class Grid, class DataStores>
                static auto make_mss_kernel(Grid const &grid, DataStores &data_stores) {
                    shared_allocator shared_alloc;

                    using plh_map_t = typename Mss::plh_map_t;
                    using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;

                    auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
                        overload(
                            [&](meta::list<cache_type::ij>, auto info) {
                                return make_ij_cache<decltype(info.data())>(
                                    info.num_colors(), IBlockSize(), JBlockSize(), info.extent(), shared_alloc);
                            },
                            [](meta::list<cache_type::k>, auto) { return k_cache_sid_t(); },
                            [&](meta::list<>, auto info) {
                                return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores));
                            }),
                        meta::transform<be_api::get_caches, plh_map_t>(),
                        plh_map_t()));

                    constexpr bool is_parallel = be_api::is_parallel<typename Mss::execution_t>{};
                    constexpr int_t k_block_size = is_parallel ? KBlockSize::value : 0;

                    auto kernel_fun = make_kernel_fun<Deref, Mss, k_block_size>(grid, composite);

                    return kernel<typename Mss::extent_t,
                        typename Mss::plh_map_t,
                        k_block_size,
                        is_parallel,
                        decltype(kernel_fun)>{std::move(kernel_fun), shared_alloc.size()};
                }

                template <class Deref,
                    template <class...>
                    class L,
                    class Grid,
                    class DataStores,
                    class PrevKernel = no_kernel>
                static void launch_msses(L<>, Grid const &grid, DataStores &&, PrevKernel prev_kernel = {}) {
                    std::move(prev_kernel).launch_or_fuse(gpu(), grid, no_kernel());
                }

                template <class Deref,
                    template <class...>
                    class L,
                    class Mss,
                    class... Msses,
                    class Grid,
                    class DataStores,
                    class PrevKernel = no_kernel>
                static void launch_msses(
                    L<Mss, Msses...>, Grid const &grid, DataStores &data_stores, PrevKernel prev_kernel = {}) {
                    auto kernel = make_mss_kernel<Deref, Mss>(grid, data_stores);
                    auto fused_kernel = std::move(prev_kernel).launch_or_fuse(gpu(), grid, std::move(kernel));
                    launch_msses<Deref>(L<Msses...>(), grid, data_stores, std::move(fused_kernel));
                }

                template <class Msses, class Grid, class DataStores>
                static void entry_point(Grid const &grid, DataStores external_data_stores) {
                    auto cuda_alloc = sid::device::cached_allocator(&cuda_util::cuda_malloc<char[]>);
                    auto data_stores = hymap::concat(
                        block(std::move(external_data_stores)), make_temporaries<Msses>(grid, cuda_alloc));
                    using const_keys_t =
                        meta::transform<be_api::get_key, meta::filter<be_api::get_is_const, typename Msses::plh_map_t>>;
                    launch_msses<deref_f<const_keys_t>>(meta::rename<meta::list, Msses>(), grid, data_stores);
                }

                template <class Spec, class Grid, class DataStores>
                friend void gridtools_backend_entry_point(gpu, Spec spec, Grid const &grid, DataStores data_stores) {
                    assert(fill_flush::validate_k_bounds<Spec>(grid, data_stores));
                    using new_spec_t = fill_flush::transform_spec<Spec>;
                    using msses_t = be_api::make_fused_view<new_spec_t>;
                    gpu::entry_point<msses_t>(
                        grid, fill_flush::transform_data_stores<typename msses_t::plh_map_t>(std::move(data_stores)));
                }
            };
        } // namespace gpu_backend
        using gpu_backend::gpu;
    } // namespace stencil
} // namespace gridtools
