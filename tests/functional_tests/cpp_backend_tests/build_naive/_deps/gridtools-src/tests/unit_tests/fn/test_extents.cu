/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/fn/extents.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools::fn {
    namespace {
        struct a;
        struct b;
        struct c;

        using ext_t = extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>;

        __device__ bool check_offsets() {
            auto testee = extend_offsets<ext_t>(hymap::keys<a, b, c>::make_values(0, 1, 2));
            return device::at_key<a>(testee) == -1 && device::at_key<b>(testee) == 1 && device::at_key<c>(testee) == 3;
        }

        TEST(extend_offsets, device) {
            EXPECT_TRUE(on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&check_offsets)));
        }

        __device__ bool check_sizes() {
            auto testee = extend_sizes<ext_t>(hymap::keys<a, b, c>::make_values(4, 5, 6));
            return device::at_key<a>(testee) == 5 && device::at_key<b>(testee) == 7 && device::at_key<c>(testee) == 6;
        }

        TEST(extend_sizes, device) { EXPECT_TRUE(on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&check_sizes))); }
    } // namespace
} // namespace gridtools::fn

#include "test_extents.cpp"
