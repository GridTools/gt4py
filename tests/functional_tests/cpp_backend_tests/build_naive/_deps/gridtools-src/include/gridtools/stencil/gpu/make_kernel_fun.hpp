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

#include <utility>

#include "../../common/defs.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../sid/blocked_dim.hpp"
#include "../../sid/concept.hpp"
#include "../be_api.hpp"
#include "../common/dim.hpp"
#include "k_cache.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            namespace make_kernel_fun_impl_ {
                GT_FUNCTION_DEVICE void syncthreads(std::true_type) { __syncthreads(); }
                GT_FUNCTION_DEVICE void syncthreads(std::false_type) {}

                template <class Deref, class Info, class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void exec_cells(
                    Info, Ptr const &ptr, Strides const &strides, Validator const &validator) {
                    device::for_each<typename Info::cells_t>([&](auto cell) GT_FORCE_INLINE_LAMBDA {
                        syncthreads(cell.need_sync());
                        if (validator(cell.extent()))
                            cell.template operator()<Deref>(ptr, strides);
                    });
                }

                template <class Deref,
                    class Mss,
                    class Sizes,
                    int_t BlockSize,
                    class = typename has_k_caches<Mss>::type>
                struct k_loop_f;

                template <class Deref, class Mss, class Sizes>
                struct k_loop_f<Deref, Mss, Sizes, 0, std::true_type> {
                    Sizes m_sizes;

                    template <class Ptr, class Strides, class Validator>
                    GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                        k_caches_type<Mss> k_caches;
                        auto mixed_ptr = hymap::device::merge(k_caches.ptr(), std::move(ptr));
                        tuple_util::device::for_each(
                            [&](const int_t size, auto info) GT_FORCE_INLINE_LAMBDA {
#ifdef __HIPCC__
// unroll factor estimate based on GT perftests on AMD Mi50
#pragma unroll 3
#else
// unroll factor estimate based on COSMO dycore performance on NVIDIA V100: no unrolling
#endif
                                for (int_t i = 0; i < size; ++i) {
                                    exec_cells<Deref>(info, mixed_ptr, strides, validator);
                                    k_caches.slide(info.k_step());
                                    info.inc_k(mixed_ptr.secondary(), strides);
                                }
                            },
                            m_sizes,
                            Mss::interval_infos());
                    }
                };

                template <class Deref, class Mss, class Sizes>
                struct k_loop_f<Deref, Mss, Sizes, 0, std::false_type> {
                    Sizes m_sizes;

                    template <class Ptr, class Strides, class Validator>
                    GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                        tuple_util::device::for_each(
                            [&](const int_t size, auto info) GT_FORCE_INLINE_LAMBDA {
#ifdef __HIPCC__
// unroll factor estimate based on GT perftests on AMD Mi50
#pragma unroll 3
#else
// unroll factor estimate based on COSMO dycore performance on NVIDIA V100: no unrolling
#endif
                                for (int_t i = 0; i < size; ++i) {
                                    exec_cells<Deref>(info, ptr, strides, validator);
                                    info.inc_k(ptr, strides);
                                }
                            },
                            m_sizes,
                            Mss::interval_infos());
                    }
                };

                template <class Deref, class Mss, class Sizes, int_t BlockSize>
                struct k_loop_f<Deref, Mss, Sizes, BlockSize, std::false_type> {
                    Sizes m_sizes;

                    template <class Ptr, class Strides, class Validator>
                    GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                        int_t cur = -(int_t)blockIdx.z * BlockSize;
                        sid::shift(ptr, sid::get_stride<dim::k>(strides), -cur);
                        tuple_util::device::for_each(
                            [&](int_t size, auto info) GT_FORCE_INLINE_LAMBDA {
                                if (cur >= BlockSize)
                                    return;
                                int_t lim = math::min(cur + size, BlockSize) - math::max(cur, 0);
                                cur += size;
#pragma unroll
                                for (int_t i = 0; i < BlockSize; ++i) {
                                    if (i >= lim)
                                        break;
                                    exec_cells<Deref>(info, ptr, strides, validator);
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
                    GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator validator) const {
                        auto ptr = m_ptr_holder();
                        sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides), blockIdx.x);
                        sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides), blockIdx.y);
                        sid::shift(ptr, sid::get_stride<dim::i>(m_strides), i_block);
                        sid::shift(ptr, sid::get_stride<dim::j>(m_strides), j_block);
                        k_loop(std::move(ptr), m_strides, std::move(validator));
                    }
                };

                template <class Deref, class Mss, int_t KBlockSize, class Grid, class Composite>
                auto make_kernel_fun(Grid const &grid, Composite &composite) {
                    sid::ptr_diff_type<Composite> offset{};
                    auto strides = sid::get_strides(composite);
                    sid::shift(
                        offset, sid::get_stride<dim::k>(strides), grid.k_start(Mss::interval(), Mss::execution()));
                    auto k_sizes = be_api::make_k_sizes(Mss::interval_infos(), grid);
                    using k_sizes_t = decltype(k_sizes);
                    using k_loop_t = k_loop_f<Deref, Mss, k_sizes_t, KBlockSize>;

                    return kernel_f<Composite, k_loop_t>{
                        sid::get_origin(composite) + offset, std::move(strides), {std::move(k_sizes)}};
                }
            } // namespace make_kernel_fun_impl_
            using make_kernel_fun_impl_::make_kernel_fun;
        } // namespace gpu_backend
    }     // namespace stencil
} // namespace gridtools
