/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/cuda_is_ptr.hpp>

#include <gtest/gtest.h>

#include <memory>

TEST(test_is_gpu_ptr, host_ptr_is_no_cuda_ptr) {
    auto testee = std::unique_ptr<double>(new double);
    EXPECT_FALSE(gridtools::is_gpu_ptr(testee.get()));
}
