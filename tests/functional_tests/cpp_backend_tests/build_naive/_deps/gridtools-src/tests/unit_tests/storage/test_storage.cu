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

__global__ void check_s1(int *s) {
    assert(s[0] == 10);
    assert(s[1] == 20);
    s[0] = 30;
    s[1] = 40;
}

__global__ void check_s2(int *s) {
    assert(s[0] == 100);
    assert(s[1] == 200);
    s[0] = 300;
    s[1] = 400;
}

TEST(StorageCudaTest, Simple) {
    auto builder = gridtools::storage::builder<gridtools::storage::gpu>.type<int>().dimensions(2);
    // create two storages
    auto s1 = builder();
    auto s2 = builder();

    // write some values
    s1->host_view()(0) = 10;
    s1->host_view()(1) = 20;
    s2->host_view()(0) = 100;
    s2->host_view()(1) = 200;
    // assert if the values were not copied correctly and reset values
    check_s1<<<1, 1>>>(s1->get_target_ptr());
    check_s2<<<1, 1>>>(s2->get_target_ptr());

    // check values
    EXPECT_EQ(s1->host_view()(1), 40);
    EXPECT_EQ(s1->host_view()(0), 30);
    EXPECT_EQ(s2->host_view()(1), 400);
    EXPECT_EQ(s2->host_view()(0), 300);
}
