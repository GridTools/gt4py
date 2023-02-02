/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/storage/builder.hpp>

#include <storage_select.hpp>

using namespace gridtools;
using testing::ElementsAre;

const auto builder = storage::builder<storage_traits_t>.type<double>();

TEST(DataStoreTest, Simple) {
    auto ds = builder.dimensions(3, 3, 3).value(5.3).build();
    auto &&info = ds->info();

    EXPECT_THAT(info.lengths(), ElementsAre(3, 3, 3));

    auto view = ds->host_view();

    EXPECT_DOUBLE_EQ(view(0, 0, 0), 5.3);
    EXPECT_DOUBLE_EQ(view(1, 1, 1), 5.3);

    view(0, 0, 0) = 100;
    view(1, 1, 1) = 200;

    EXPECT_DOUBLE_EQ(view(0, 0, 0), 100);
    EXPECT_DOUBLE_EQ(view(1, 1, 1), 200);
}

TEST(DataStoreTest, Initializer) {
    auto ds = builder.dimensions(128, 128, 80).value(3.1415).build();
    auto lengths = ds->lengths();
    auto view = ds->host_view();
    for (uint_t i = 0; i < lengths[0]; ++i)
        for (uint_t j = 0; j < lengths[1]; ++j)
            for (uint_t k = 0; k < lengths[2]; ++k)
                EXPECT_DOUBLE_EQ(view(i, j, k), 3.1415);
}

TEST(DataStoreTest, LambdaInitializer) {
    auto ds = builder.dimensions(10, 11, 12).initializer([](int i, int j, int k) { return i + j + k; }).build();
    auto lengths = ds->lengths();
    auto view = ds->host_view();
    for (uint_t i = 0; i < lengths[0]; ++i)
        for (uint_t j = 0; j < lengths[1]; ++j)
            for (uint_t k = 0; k < lengths[2]; ++k)
                EXPECT_DOUBLE_EQ(view(i, j, k), i + j + k);
}

TEST(DataStoreTest, Naming) {
    auto builder = ::builder.dimensions(10, 11, 12);
    // no naming
    auto ds2_nn = builder();
    auto ds3_nn = builder.value(1)();
    auto ds4_nn = builder.initializer([](int i, int j, int k) { return i + j + k; })();
    EXPECT_EQ(ds2_nn->name(), "");
    EXPECT_EQ(ds3_nn->name(), "");
    EXPECT_EQ(ds4_nn->name(), "");

    // test naming
    auto ds2 = builder.name("standard storage")();
    auto ds3 = builder.value(1).name("value init. storage")();
    auto ds4 = builder.initializer([](int i, int j, int k) { return i + j + k; }).name("lambda init. storage")();
    EXPECT_EQ(ds2->name(), "standard storage");
    EXPECT_EQ(ds3->name(), "value init. storage");
    EXPECT_EQ(ds4->name(), "lambda init. storage");
}

TEST(DataStoreTest, DimAndSizeInterface) {
    auto ds = builder.dimensions(128, 128, 80)();
    EXPECT_THAT(ds->lengths(), ElementsAre(128, 128, 80));
}
