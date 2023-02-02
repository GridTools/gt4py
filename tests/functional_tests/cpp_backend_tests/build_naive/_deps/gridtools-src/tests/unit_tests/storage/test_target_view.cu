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

#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/gpu.hpp>

#include <multiplet.hpp>

using namespace gridtools;

constexpr int c_x = 3 /* < 32 for this test */, c_y = 5, c_z = 7;

template <class View>
__global__ void mul2(View s) {
    auto &&lengths = s.lengths();
    bool expected_dims = lengths[0] == c_x && lengths[1] == c_y && lengths[2] == c_z;
    bool expected_size = s.length() <= 32 * c_y * c_z && s.length() >= c_x * c_y * c_z;
    s(0, 0, 0) *= 2 * expected_dims * expected_size;
    s(1, 0, 0) *= 2 * expected_dims * expected_size;
}

TEST(DataViewTest, Simple) {
    // create and allocate a data_store
    auto ds = storage::builder<storage::gpu>.type<double>().layout<2, 1, 0>().dimensions(c_x, c_y, c_z)();
    // create a rw view and fill with some data
    auto dv = ds->host_view();
    dv(0, 0, 0) = 50;
    dv(1, 0, 0) = 60;

    // check if interface works
    EXPECT_TRUE(ds->lengths() == dv.lengths());

    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(60, dv(1, 0, 0));
    // create a ro view
    auto dvro = ds->const_host_view();
    // check if data is the same
    EXPECT_EQ(50, dvro(0, 0, 0));
    EXPECT_EQ(60, dvro(1, 0, 0));

    mul2<<<1, 1>>>(ds->target_view());

    dvro = ds->const_host_view();
    // check if data is the same
    EXPECT_EQ(100, dvro(0, 0, 0));
    EXPECT_EQ(120, dvro(1, 0, 0));
}
