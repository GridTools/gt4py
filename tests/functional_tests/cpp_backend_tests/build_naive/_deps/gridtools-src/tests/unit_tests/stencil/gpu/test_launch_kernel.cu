/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/gpu/launch_kernel.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/stencil/common/extent.hpp>

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            template <class Extent, int_t IBlockSize, int_t JBlockSize>
            struct validation_kernel_f {
                int *m_failures;
                int_t m_i_size;
                int_t m_j_size;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t iblock, int_t jblock, Validator is_valid) const {
                    int_t i_block_size =
                        (blockIdx.x + 1) * IBlockSize < m_i_size ? IBlockSize : m_i_size - blockIdx.x * IBlockSize;
                    int_t j_block_size =
                        (blockIdx.y + 1) * JBlockSize < m_j_size ? JBlockSize : m_j_size - blockIdx.y * JBlockSize;
                    bool expected = Extent::iminus::value <= iblock && Extent::iplus::value + i_block_size > iblock &&
                                    Extent::jminus::value <= jblock && Extent::jplus::value + j_block_size > jblock;
                    bool actual = is_valid(Extent());
                    if (actual == expected)
                        return;
                    atomicAdd(m_failures, 1);
                    int block_idx_x = blockIdx.x;
                    int block_idx_y = blockIdx.y;
                    printf("false %s at {%d,%d} of block {%d,%d}\n",
                        actual ? "positive" : "negative",
                        iblock,
                        jblock,
                        block_idx_x,
                        block_idx_y);
                }
            };

            template <class MaxExtent, class Extent, int_t IBlockSize, int_t JBlockSize>
            void do_validation_test(int_t i_size, int_t j_size) {
                auto failures = cuda_util::make_clone(0);
                validation_kernel_f<Extent, IBlockSize, JBlockSize> kernel = {failures.get(), i_size, j_size};
                launch_kernel<MaxExtent, IBlockSize, JBlockSize>(i_size, j_size, 1, kernel, 0);
                EXPECT_EQ(0, cuda_util::from_clone(failures));
            }

            TEST(validation, simplest) { do_validation_test<extent<>, extent<>, 32, 8>(128, 128); }

            TEST(validation, rounded_sizes) {
                do_validation_test<extent<-2, 2, -1, 3>, extent<-1, 1, 0, 2>, 32, 8>(128, 128);
            }

            TEST(validation, hori_diff) {
                do_validation_test<extent<-1, 1, -1, 1>, extent<-1, 1, -1, 1>, 32, 8>(128, 128);
            }

            TEST(validation, hori_diff_small_size) {
                do_validation_test<extent<-1, 1, -1, 1>, extent<-1, 1, -1, 1>, 32, 8>(5, 5);
            }

            TEST(validation, max_extent) {
                do_validation_test<extent<-2, 2, -1, 3>, extent<-2, 2, -1, 3>, 32, 8>(123, 50);
            }

            TEST(validation, zero_extent) { do_validation_test<extent<-2, 2, -1, 3>, extent<>, 32, 8>(123, 50); }

            TEST(validation, reduced_extent) {
                do_validation_test<extent<-2, 2, -1, 3>, extent<-1, 1, 0, 2>, 32, 8>(123, 50);
            }

            struct syncthreads_kernel_f {
                int *m_failures;
                int *m_count;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t iblock, int_t jblock, Validator is_valid) const {
                    if (is_valid(extent<-1, 1>())) {
                        assert(jblock == 0);
                        assert(iblock >= -1 && iblock <= 1);
                        atomicAdd(m_count, 1);
                    }
                    __syncthreads();
                    if (is_valid(extent<-1, 1>())) {
                        assert(jblock == 0);
                        assert(iblock >= -1 && iblock <= 1);
                        auto count = atomicAdd(m_count, 0);
                        if (count == 3)
                            return;
                        atomicAdd(m_failures, 1);
                        printf("failure: i = %d, count == %d\n", iblock, count);
                    }
                }
            };

            TEST(syncthreads, smoke) {
                auto failures = cuda_util::make_clone(0);
                auto count = cuda_util::make_clone(0);
                launch_kernel<extent<-1, 1>, 32, 8>(1, 1, 1, syncthreads_kernel_f{failures.get(), count.get()}, 0);
                EXPECT_EQ(0, cuda_util::from_clone(failures));
            }
        } // namespace gpu_backend
    }     // namespace stencil
} // namespace gridtools
