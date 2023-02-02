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

#include <multiplet.hpp>
#include <storage_select.hpp>

using namespace gridtools;

const auto builder = storage::builder<storage_traits_t>.type<double>();

TEST(DataViewTest, Simple) {
    auto builder = ::builder.dimensions(3, 5, 7);
    // create and allocate a data_store
    auto ds = builder();
    // create a rw view and fill with some data
    auto dv = ds->host_view();
    dv(0, 0, 0) = 50;
    dv(0, 0, 1) = 60;

    // check if interface works
    EXPECT_TRUE(ds->lengths() == dv.lengths());

    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(dv(0, 0, 1), 60);

    // create a ro view
    auto dvro = ds->const_host_view();
    // check if data is the same
    EXPECT_EQ(50, dvro(0, 0, 0));
    EXPECT_EQ(dvro(0, 0, 1), 60);

    // create  a second storage
    auto ds_tmp = builder();
    // again create a view
    auto dv_tmp = ds_tmp->const_host_view();
}

TEST(DataViewTest, ArrayAPI) {
    // create and allocate a data_store
    auto ds = builder.dimensions(2, 2, 2)();
    auto dvro = ds->host_view();

    dvro({1, 1, 1}) = 2.0;
    EXPECT_TRUE((dvro({1, 1, 1}) == 2.0));
}

TEST(DataViewTest, Looping) {
    auto ds = storage::builder<storage_traits_t >.type<triplet>().name("ds")
                  .dimensions(2 + 2, 2 + 4, 2 + 6)
                  .halos(1, 2, 3)
                  .initializer([](int i, int j, int k) {
                      return triplet{i, j, k};
                  })
                  .build();

    auto view = ds->const_host_view();

    auto &&lengths = view.lengths();
    for (int i = 0; i < lengths[0]; ++i)
        for (int j = 0; j < lengths[1]; ++j)
            for (int k = 0; k < lengths[2]; ++k)
                EXPECT_EQ(view(i, j, k), (triplet{i, j, k}));
}
