/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/sid/allocator.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools {
    namespace {

        template <typename PtrHolder>
        __device__ bool check_allocation(PtrHolder ptr_holder) {
            auto &ref = *ptr_holder();
            ref = 1.;
            return ref == 1.;
        }

        template <typename PtrHolder>
        __global__ void test_allocated(PtrHolder testee, bool *result) {}

        TEST(simple_device_memory_allocator, test) {
            sid::device::allocator<GT_INTEGRAL_CONSTANT_FROM_VALUE(&cuda_util::cuda_malloc<char[]>)> alloc;
            auto ptr_holder = allocate(alloc, meta::lazy::id<double>{}, 1);

            auto result = gridtools::on_device::exec(
                GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&check_allocation<decltype(ptr_holder)>), ptr_holder);
            ASSERT_TRUE(result);
        }
    } // namespace
} // namespace gridtools
