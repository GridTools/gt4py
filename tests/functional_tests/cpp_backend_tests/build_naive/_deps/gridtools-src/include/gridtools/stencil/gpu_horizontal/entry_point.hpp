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

#include "../../common/cuda_type_traits.hpp"
#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/functional.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../sid/as_const.hpp"
#include "../../sid/block.hpp"
#include "../../sid/blocked_dim.hpp"
#include "../../sid/composite.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/sid_shift_origin.hpp"
#include "../be_api.hpp"
#include "../common/caches.hpp"
#include "../common/dim.hpp"
#include "../common/extent.hpp"
#include "j_cache.hpp"
#include "launch_kernel.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_horizontal_backend {
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

            template <int_t JBlockSize>
            struct j_loop_f {
                template <class Info, class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(
                    Info, Ptr ptr, Strides const &strides, Validator const &validator) const {
                    using namespace literals;
                    constexpr auto step = 1_c;
                    using max_extent_t = typename Info::extent_t;
                    using const_keys_t =
                        meta::transform<be_api::get_key, meta::filter<be_api::get_is_const, typename Info::plh_map_t>>;

                    constexpr auto j_start = typename max_extent_t::jminus() - typename max_extent_t::jplus();
                    sid::shift(ptr, sid::get_stride<dim::j>(strides), j_start);

                    using j_caches_t = j_caches_type<Info>;
                    j_caches_t j_caches;
                    auto mixed_ptr = hymap::device::merge(j_caches.ptr(), std::move(ptr));

                    auto shift_mixed_ptr = [&](auto dim, auto offset) {
                        sid::shift(mixed_ptr.secondary(), sid::get_stride<decltype(dim)>(strides), offset);
                        j_caches_t::shift(dim, mixed_ptr.primary(), offset);
                    };

#pragma unroll
                    for (int_t j = decltype(j_start)::value; j < JBlockSize; ++j) {
                        device::for_each<typename Info::cells_t>([&](auto cell) GT_FORCE_INLINE_LAMBDA {
                            using cell_t = decltype(cell);
                            using extent_t = typename cell_t::extent_t;
                            constexpr auto j_offset = typename extent_t::jplus();
                            shift_mixed_ptr(dim::j(), j_offset);
                            shift_mixed_ptr(dim::i(), typename extent_t::iminus());
#pragma unroll
                            for (int_t i = extent_t::iminus::value; i <= extent_t::iplus::value; ++i) {
                                if (validator(extent_t(), i, j + j_offset))
                                    cell.template operator()<deref_f<const_keys_t>>(mixed_ptr, strides);
                                shift_mixed_ptr(dim::i(), step);
                            }
                            shift_mixed_ptr(dim::i(), -typename extent_t::iplus() - step);
                            shift_mixed_ptr(dim::j(), -j_offset);
                        });
                        sid::shift(mixed_ptr.secondary(), sid::get_stride<dim::j>(strides), step);
                        j_caches.slide();
                    }
                }
            };

            template <class JLoop, class Mss, class Sizes, int_t KBlockSize>
            struct k_loop_f {
                Sizes m_sizes;

                template <class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                    int_t cur = -(int_t)blockIdx.z * KBlockSize;
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), -cur);
                    tuple_util::device::for_each(
                        [&](int_t size, auto info) GT_FORCE_INLINE_LAMBDA {
                            if (cur >= KBlockSize)
                                return;
                            int_t lim = math::min(cur + size, KBlockSize) - math::max(cur, 0);
                            cur += size;
#pragma unroll
                            for (int_t i = 0; i < KBlockSize; ++i) {
                                if (i >= lim)
                                    break;
                                JLoop()(info, ptr, strides, validator);
                                info.inc_k(ptr, strides);
                            }
                        },
                        m_sizes,
                        Mss::interval_infos());
                }
            };

            template <class Sid, class KLoop>
            struct kernel_f {
                sid::ptr_holder_type<Sid> m_ptr_holder;
                sid::strides_type<Sid> m_strides;
                KLoop k_loop;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t i_block, Validator validator) const {
                    auto ptr = m_ptr_holder();
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides), blockIdx.x);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides), blockIdx.y);
                    sid::shift(ptr, sid::get_stride<dim::i>(m_strides), i_block);
                    k_loop(std::move(ptr), m_strides, std::move(validator));
                }
            };

            template <class IBlockSize = integral_constant<int_t, 256>,
                class JBlockSize = integral_constant<int_t, 8>,
                class KBlockSize = integral_constant<int_t, 1>>
            struct gpu_horizontal {
                template <class Mss, class Grid, class Composite>
                static auto make_kernel_fun(Grid const &grid, Composite &composite) {
                    sid::ptr_diff_type<Composite> offset{};
                    auto strides = sid::get_strides(composite);
                    sid::shift(
                        offset, sid::get_stride<dim::k>(strides), grid.k_start(Mss::interval(), Mss::execution()));
                    auto k_sizes = be_api::make_k_sizes(Mss::interval_infos(), grid);
                    using k_sizes_t = decltype(k_sizes);
                    using k_loop_t = k_loop_f<j_loop_f<JBlockSize::value>, Mss, k_sizes_t, KBlockSize::value>;
                    return kernel_f<Composite, k_loop_t>{
                        sid::get_origin(composite) + offset, std::move(strides), {std::move(k_sizes)}};
                }

                template <class Spec, class Grid, class DataStores>
                static void entry_point(Grid const &grid, DataStores data_stores) {
                    using msses_t = be_api::make_fused_view<Spec>;
                    static_assert(meta::length<msses_t>::value == 1, "Not implemented");
                    using mss_t = meta::first<msses_t>;
                    static_assert(be_api::is_parallel<typename mss_t::execution_t>(), "Not implemented");
                    using plh_map_t = typename mss_t::plh_map_t;

                    using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;
                    auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
                        overload( //
                            [&](std::true_type, auto info) { return j_cache_sid_t<decltype(info)>(); },
                            [&](std::false_type, auto info) {
                                return sid::add_const(info.is_const(),
                                    sid::block(at_key<decltype(info.plh())>(data_stores),
                                        hymap::keys<dim::i, dim::j>::make_values(IBlockSize(), JBlockSize())));
                            }),
                        meta::transform<be_api::get_is_tmp, plh_map_t>(),
                        plh_map_t()));

                    launch_kernel<IBlockSize::value, JBlockSize::value>(grid.i_size(),
                        grid.j_size(),
                        (grid.k_size() + KBlockSize::value - 1) / KBlockSize::value,
                        make_kernel_fun<mss_t>(grid, composite));
                }

                template <class Spec, class Grid, class DataStores>
                friend void gridtools_backend_entry_point(
                    gpu_horizontal, Spec, Grid const &grid, DataStores data_stores) {
                    return gpu_horizontal::entry_point<Spec>(grid, std::move(data_stores));
                }
            };
        } // namespace gpu_horizontal_backend
        using gpu_horizontal_backend::gpu_horizontal;
    } // namespace stencil
} // namespace gridtools
