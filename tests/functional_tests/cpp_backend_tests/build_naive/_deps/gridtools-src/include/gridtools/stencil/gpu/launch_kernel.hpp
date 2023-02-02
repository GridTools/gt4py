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

#include "../../common/cuda_runtime.hpp"
#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../common/extent.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            /*
             *  In a typical cuda block we have the following regions
             *
             *    aa bbbbbbbb cc
             *    aa bbbbbbbb cc
             *
             *    hh dddddddd ii
             *    hh dddddddd ii
             *    hh dddddddd ii
             *    hh dddddddd ii
             *
             *    ee ffffffff gg
             *    ee ffffffff gg
             *
             * Regions b,d,f have warp (or multiple of warp size)
             * Size of regions a, c, h, i, e, g are determined by max_extent_t
             * Regions b,d,f are easily executed by dedicated warps (one warp for each line).
             * Regions (a,h,e) and (c,i,g) are executed by two specialized warp
             */

            namespace launch_kernel_impl_ {

                constexpr int_t ceil(int_t x) { return x < 2 ? 1 : 2 * ceil((x + 1) / 2); }

                template <class MaxExtent>
                struct extent_validator_f {
                    static_assert(is_extent<MaxExtent>::value, GT_INTERNAL_ERROR);

                    int_t m_i_lo;
                    int_t m_i_hi;
                    int_t m_j_lo;
                    int_t m_j_hi;

                    GT_FUNCTION_DEVICE extent_validator_f(
                        int_t i_pos, int_t j_pos, int_t i_block_size, int_t j_block_size)
                        : m_i_lo(i_pos), m_i_hi(i_pos - i_block_size), m_j_lo(j_pos), m_j_hi(j_pos - j_block_size) {}

                    template <class Extent = MaxExtent>
                    GT_FUNCTION_DEVICE bool operator()(Extent = {}) const {
                        static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                        static_assert(Extent::iminus::value >= MaxExtent::iminus::value, GT_INTERNAL_ERROR);
                        static_assert(Extent::iplus::value <= MaxExtent::iplus::value, GT_INTERNAL_ERROR);
                        static_assert(Extent::jminus::value >= MaxExtent::jminus::value, GT_INTERNAL_ERROR);
                        static_assert(Extent::jplus::value <= MaxExtent::jplus::value, GT_INTERNAL_ERROR);

                        return Extent::iminus::value <= m_i_lo && Extent::iplus::value > m_i_hi &&
                               Extent::jminus::value <= m_j_lo && Extent::jplus::value > m_j_hi;
                    }
                };

                struct dummy_validator_f {
                    template <class Extent = extent<>>
                    GT_FUNCTION_DEVICE bool operator()(Extent = {}) const {
                        static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                        return true;
                    }
                };

                template <size_t NumThreads, int_t BlockSizeI, int_t BlockSizeJ, class Extent, class Fun>
                __global__ void __launch_bounds__(NumThreads) wrapper(Fun const fun, int_t i_size, int_t j_size) {
                    // jboundary_limit determines the number of warps required to execute (b,d,f)
                    static constexpr auto jboundary_limit = BlockSizeJ + Extent::jplus::value - Extent::jminus::value;
                    // iminus_limit adds to jboundary_limit an additional warp for regions (a,h,e)
                    static constexpr auto iminus_limit = jboundary_limit + (Extent::iminus::value < 0 ? 1 : 0);

                    int_t i_block, j_block;

                    if (threadIdx.y < jboundary_limit) {
                        i_block = (int_t)threadIdx.x;
                        j_block = (int_t)threadIdx.y + Extent::jminus::value;
                    } else if (threadIdx.y < iminus_limit) {
                        assert(Extent::iminus::value < 0);
                        static constexpr auto boundary = ceil(-Extent::iminus::value);
                        // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough threads
                        static_assert(jboundary_limit * boundary <= BlockSizeI, GT_INTERNAL_ERROR);

                        i_block = -boundary + (int_t)threadIdx.x % boundary;
                        j_block = (int_t)threadIdx.x / boundary + Extent::jminus::value;
                    } else {
                        assert(Extent::iplus::value > 0);
                        assert(threadIdx.y < iminus_limit + 1);
                        static constexpr auto boundary = ceil(Extent::iplus::value);
                        // we dedicate one warp to execute regions (c,i,g), so here we make sure we have enough threads
                        static_assert(jboundary_limit * boundary <= BlockSizeI, GT_INTERNAL_ERROR);

                        i_block = (int_t)threadIdx.x % boundary + BlockSizeI;
                        j_block = (int_t)threadIdx.x / boundary + Extent::jminus::value;
                    }

                    int_t i_block_size =
                        (blockIdx.x + 1) * BlockSizeI < i_size ? BlockSizeI : i_size - blockIdx.x * BlockSizeI;
                    int_t j_block_size =
                        (blockIdx.y + 1) * BlockSizeJ < j_size ? BlockSizeJ : j_size - blockIdx.y * BlockSizeJ;

                    fun(i_block, j_block, extent_validator_f<Extent>{i_block, j_block, i_block_size, j_block_size});
                }

                template <size_t NumThreads, int_t BlockSizeI, int_t BlockSizeJ, class Fun>
                __global__ void __launch_bounds__(NumThreads)
                    zero_extent_wrapper(Fun const fun, int_t i_size, int_t j_size) {
                    if (blockIdx.x * BlockSizeI + threadIdx.x < i_size &&
                        blockIdx.y * BlockSizeJ + threadIdx.y < j_size)
                        fun(threadIdx.x, threadIdx.y, dummy_validator_f());
                }

                template <class Extent>
                constexpr bool is_empty_ij_extents() {
                    return Extent::iminus::value == 0 && Extent::iplus::value == 0 && Extent::jminus::value == 0 &&
                           Extent::jplus::value == 0;
                }

                template <class Extent,
                    int_t BlockSizeI,
                    int_t BlockSizeJ,
                    class Fun,
                    std::enable_if_t<!is_empty_ij_extents<Extent>(), int> = 0>
                void launch_kernel(int_t i_size, int_t j_size, uint_t zblocks, Fun fun, size_t shared_memory_size = 0) {
                    static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                    static_assert(Extent::iminus::value <= 0, GT_INTERNAL_ERROR);
                    static_assert(Extent::iplus::value >= 0, GT_INTERNAL_ERROR);

                    static_assert(std::is_trivially_copy_constructible_v<Fun>, GT_INTERNAL_ERROR);

                    static constexpr auto halo_lines = Extent::jplus::value - Extent::jminus::value +
                                                       (Extent::iminus::value < 0 ? 1 : 0) +
                                                       (Extent::iplus::value > 0 ? 1 : 0);
                    static constexpr size_t num_threads = BlockSizeI * (BlockSizeJ + halo_lines);

                    uint_t xblocks = (i_size + BlockSizeI - 1) / BlockSizeI;
                    uint_t yblocks = (j_size + BlockSizeJ - 1) / BlockSizeJ;

                    dim3 blocks = {xblocks, yblocks, zblocks};
                    dim3 threads = {BlockSizeI, BlockSizeJ + halo_lines, 1};

                    cuda_util::launch(blocks,
                        threads,
                        shared_memory_size,
                        0, // if required propagate CUDA stream
                        wrapper<num_threads, BlockSizeI, BlockSizeJ, Extent, Fun>,
                        std::move(fun),
                        i_size,
                        j_size);
                }

                template <class Extent,
                    int_t BlockSizeI,
                    int_t BlockSizeJ,
                    class Fun,
                    std::enable_if_t<is_empty_ij_extents<Extent>(), int> = 0>
                void launch_kernel(int_t i_size, int_t j_size, uint_t zblocks, Fun fun, size_t shared_memory_size = 0) {

                    static_assert(std::is_trivially_copy_constructible_v<Fun>, GT_INTERNAL_ERROR);

                    static const size_t num_threads = BlockSizeI * BlockSizeJ;

                    uint_t xblocks = (i_size + BlockSizeI - 1) / BlockSizeI;
                    uint_t yblocks = (j_size + BlockSizeJ - 1) / BlockSizeJ;

                    dim3 blocks = {xblocks, yblocks, zblocks};
                    dim3 threads = {BlockSizeI, BlockSizeJ, 1};

                    cuda_util::launch(blocks,
                        threads,
                        shared_memory_size,
                        0, // if required propagate CUDA stream
                        zero_extent_wrapper<num_threads, BlockSizeI, BlockSizeJ, Fun>,
                        std::move(fun),
                        i_size,
                        j_size);
                }
            } // namespace launch_kernel_impl_

            /**
             * Launch the functor `fun` with the signature `fun(int_t i_pos, int_t j_pos, ExtentValidator validator)`
             * as a cuda kernel.
             *
             * The functor should be callable in device.
             * `i_pos` parameter refers to the i position within the logical block. Note that it can go out of the block
             * bounds. The user can check if the position is valid for the given extent with the provided validator:
             *
             * struct fun_f {
             *  __device__ void fun(int i, int j, Validator v) const {
             *    if (v(extent<-1, 1>())) {
             *       // do stuff
             *    }
             *  }
             * };
             *
             *  \param Extent maximum requested extent
             *  \param BlockSizeI the logical size of the block in i dimension
             *  \param BlockSizeJ the logical size of the block in j dimension
             *  \param i_size the size of the computation area in i dimension
             *  \param j_size the size of the computation area in j dimension
             *  \param zblocks number of the blocks in k dimension
             *  \param shared_memory_size delegated to cuda
             */
            using launch_kernel_impl_::launch_kernel;
        } // namespace gpu_backend
    }     // namespace stencil
} // namespace gridtools
