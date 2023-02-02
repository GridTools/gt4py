/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/gpu/shared_allocator.hpp>

#include <gtest/gtest.h>

#include <gridtools/meta.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            namespace {
                template <typename PtrHolder>
                __device__ uint64_t get_ptr(PtrHolder ptr_holder) {
                    return reinterpret_cast<uint64_t>(ptr_holder());
                }

                TEST(shared_allocator, alignment) {
                    shared_allocator allocator;
                    EXPECT_EQ(0, allocator.size());

                    using alloc1_t = char[14];
                    using alloc2_t = double;
                    using alloc3_t = double;

                    auto alloc1 = allocate(allocator, meta::lazy::id<alloc1_t>{}, 7);
                    auto alloc2 = allocate(allocator, meta::lazy::id<alloc2_t>{}, 4);
                    auto alloc3 = allocate(allocator, meta::lazy::id<alloc3_t>{}, 1);

                    auto ptr1 = on_device::exec_with_shared_memory(
                        allocator.size(), GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&get_ptr<decltype(alloc1)>), alloc1);
                    auto ptr2 = on_device::exec_with_shared_memory(
                        allocator.size(), GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&get_ptr<decltype(alloc2)>), alloc2);
                    auto ptr3 = on_device::exec_with_shared_memory(
                        allocator.size(), GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&get_ptr<decltype(alloc3)>), alloc3);

                    // check alignment for all allocations
                    EXPECT_EQ(ptr1 % alignof(alloc1_t), 0);
                    EXPECT_EQ(ptr2 % alignof(alloc2_t), 0);
                    EXPECT_EQ(ptr3 % alignof(alloc3_t), 0);
                }

                template <class PtrHolderFloat, class PtrHolderInt>
                __global__ void fill_and_check_test(
                    PtrHolderFloat holder1, PtrHolderFloat holder1_shifted, PtrHolderInt holder2, bool *result) {
                    static_assert(std::is_same_v<decltype(holder1()), float *>);
                    static_assert(std::is_same_v<decltype(holder1_shifted()), float *>);
                    static_assert(std::is_same_v<decltype(holder2()), int16_t *>);

                    auto ptr1 = holder1();
                    auto ptr1_shifted = holder1_shifted();
                    auto ptr2 = holder2();

                    ptr1[threadIdx.x] = 100 * blockIdx.x + threadIdx.x;
                    ptr1_shifted[threadIdx.x] = 10000 + 100 * blockIdx.x + threadIdx.x;
                    ptr2[threadIdx.x] = 20000 + 100 * blockIdx.x + threadIdx.x;
                    __syncthreads();

                    if (threadIdx.x == 0) {
                        bool local_result = true;
                        for (int i = 0; i < 32; ++i) {
                            local_result &=
                                (ptr1[i] == 100 * blockIdx.x + i && ptr1[i + 32] == 10000 + 100 * blockIdx.x + i &&
                                    ptr2[i] == 20000 + 100 * blockIdx.x + i);
                        }

                        result[blockIdx.x] = local_result;
                    }
                }

                TEST(shared_allocator, fill_and_check) {
                    shared_allocator allocator;
                    auto float_ptr = allocate(allocator, meta::lazy::id<float>{}, 64);
                    auto int_ptr = allocate(allocator, meta::lazy::id<int16_t>{}, 32);

                    bool *result;
                    GT_CUDA_CHECK(cudaMallocManaged(&result, 2 * sizeof(bool)));

                    fill_and_check_test<<<2, 32, allocator.size()>>>(
                        float_ptr, (float_ptr + 48) + (-16), int_ptr, result);
                    GT_CUDA_CHECK(cudaDeviceSynchronize());

                    EXPECT_TRUE(result[0]);
                    EXPECT_TRUE(result[1]);

                    GT_CUDA_CHECK(cudaFree(result));
                }
            } // namespace
        }     // namespace gpu_backend
    }         // namespace stencil
} // namespace gridtools
