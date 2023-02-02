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

#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../common/extent.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_horizontal_backend {
            namespace launch_kernel_impl_ {
                struct extent_validator_f {

                    int_t m_i_lo;
                    int_t m_i_hi;
                    int_t m_j_block_size;

                    GT_FUNCTION_DEVICE extent_validator_f(int_t i_pos, int_t i_block_size, int_t j_block_size)
                        : m_i_lo(i_pos), m_i_hi(i_pos - i_block_size), m_j_block_size(j_block_size) {}

                    template <class Extent, class I>
                    GT_FUNCTION_DEVICE bool operator()(Extent, I i, int_t j) const {
                        static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                        return Extent::iminus::value - i <= m_i_lo && Extent::iplus::value - i > m_i_hi &&
                               Extent::jminus::value <= j && Extent::jplus::value > j - m_j_block_size;
                    }
                };

                template <size_t NumThreads, int_t BlockSizeI, int_t BlockSizeJ, class Fun>
                __global__ void __launch_bounds__(NumThreads) wrapper(Fun const fun, int_t i_size, int_t j_size) {
                    int_t i_block = threadIdx.x;

                    int_t i_block_size =
                        (blockIdx.x + 1) * BlockSizeI < i_size ? BlockSizeI : i_size - blockIdx.x * BlockSizeI;
                    if (i_block >= i_block_size)
                        return;

                    int_t j_block_size =
                        (blockIdx.y + 1) * BlockSizeJ < j_size ? BlockSizeJ : j_size - blockIdx.y * BlockSizeJ;

                    fun(i_block, extent_validator_f{i_block, i_block_size, j_block_size});
                }

                template <int_t BlockSizeI, int_t BlockSizeJ, class Fun>
                void launch_kernel(int_t i_size, int_t j_size, uint_t zblocks, Fun fun) {
                    static_assert(std::is_trivially_copy_constructible_v<Fun>, GT_INTERNAL_ERROR);

                    cuda_util::launch(
                        dim3((i_size + BlockSizeI - 1) / BlockSizeI, (j_size + BlockSizeJ - 1) / BlockSizeJ, zblocks),
                        dim3(BlockSizeI, 1, 1),
                        0,
                        0, // if required propagate CUDA stream
                        wrapper<BlockSizeI, BlockSizeI, BlockSizeJ, Fun>,
                        std::move(fun),
                        i_size,
                        j_size);
                }
            } // namespace launch_kernel_impl_

            using launch_kernel_impl_::launch_kernel;
        } // namespace gpu_horizontal_backend
    }     // namespace stencil
} // namespace gridtools
