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

#include <type_traits>

#include <gridtools/common/defs.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/traits.hpp>

#include <storage_select.hpp>

using namespace gridtools;

template <class View>
#ifdef GT_STORAGE_GPU
__global__
#endif
    void
    computation(View v) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                v(i, j, k) *= 2;
}

auto builder = storage::builder<storage_traits_t>.type<double>();

TEST(StorageFacility, ViewTests) {
    // create a data store
    auto ds = builder.dimensions(3, 3, 3).build();

    // fill with values
    auto hv = ds->host_view();
    uint_t x = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                hv(i, j, k) = x++;

    // do some computation
    computation
#ifdef GT_STORAGE_GPU
        <<<1, 1>>>
#endif
        (ds->target_view());

    // create a read only data view
    auto hrv = ds->const_host_view();

    // validate
    uint_t z = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_EQ(hrv(i, j, k), 2 * z++);
}

template <int... Args>
static constexpr bool expect_layout =
    std::is_same_v<typename decltype(builder.dimensions(Args...)())::element_type::layout_t, layout_map<Args...>>;

template <class Layout, int... Args>
static constexpr bool expect_special_layout =
    std::is_same_v<typename decltype(builder.selector<Args...>().dimensions(Args...)())::element_type::layout_t,
        Layout>;

template <int... Args>
static constexpr bool expect_custom_layout =
    std::is_same_v<typename decltype(builder.layout<Args...>().dimensions(Args...)())::element_type::layout_t,
        layout_map<Args...>>;

static_assert(expect_custom_layout<2, 1, 0>);
static_assert(expect_custom_layout<1, 0>);
static_assert(expect_custom_layout<2, -1, 1, 0>);

#if defined(GT_STORAGE_CPU_KFIRST)

static_assert(expect_layout<0>);
static_assert(expect_layout<0, 1>);
static_assert(expect_layout<0, 1, 2>);
static_assert(expect_layout<1, 2, 3, 0>);
static_assert(expect_layout<2, 3, 4, 0, 1>);

static_assert(expect_special_layout<layout_map<2, 3, 4, 0, 1>, 1, 1, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<-1, 2, 3, 0, 1>, 0, 1, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<2, -1, 3, 0, 1>, 1, 0, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<2, 3, -1, 0, 1>, 1, 1, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<1, 2, 3, -1, 0>, 1, 1, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<1, 2, 3, 0, -1>, 1, 1, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, -1, 2, 0, 1>, 0, 0, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<2, -1, -1, 0, 1>, 1, 0, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<1, 2, -1, -1, 0>, 1, 1, 0, 0, 1>);
static_assert(expect_special_layout<layout_map<0, 1, 2, -1, -1>, 1, 1, 1, 0, 0>);

static_assert(expect_special_layout<layout_map<-1, 2, -1, 0, 1>, 0, 1, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<1, -1, 2, -1, 0>, 1, 0, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<1, 2, -1, 0, -1>, 1, 1, 0, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, 1, 2, -1, 0>, 0, 1, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<1, -1, 2, 0, -1>, 1, 0, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, 1, 2, 0, -1>, 0, 1, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, -1, -1, 0, 1>, 0, 0, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<1, -1, -1, -1, 0>, 1, 0, 0, 0, 1>);
static_assert(expect_special_layout<layout_map<0, 1, -1, -1, -1>, 1, 1, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, 0, 1, -1, -1>, 0, 1, 1, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, 1, 0, -1>, 0, 0, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<0, -1, -1, -1, -1>, 1, 0, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, 0, -1, -1, -1>, 0, 1, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, 0, -1, -1>, 0, 0, 1, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, -1, 0, -1>, 0, 0, 0, 1, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, -1, -1, 0>, 0, 0, 0, 0, 1>);

#elif defined(GT_STORAGE_CPU_IFIRST)

static_assert(expect_layout<0>);
static_assert(expect_layout<1, 0>);
static_assert(expect_layout<2, 0, 1>);
static_assert(expect_layout<3, 1, 2, 0>);
static_assert(expect_layout<4, 2, 3, 1, 0>);

static_assert(expect_special_layout<layout_map<4, 2, 3, 1, 0>, 1, 1, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<-1, 2, 3, 1, 0>, 0, 1, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<3, -1, 2, 1, 0>, 1, 0, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<3, 2, -1, 1, 0>, 1, 1, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<3, 1, 2, -1, 0>, 1, 1, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<3, 1, 2, 0, -1>, 1, 1, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, -1, 2, 1, 0>, 0, 0, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<2, -1, -1, 1, 0>, 1, 0, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<2, 1, -1, -1, 0>, 1, 1, 0, 0, 1>);
static_assert(expect_special_layout<layout_map<2, 0, 1, -1, -1>, 1, 1, 1, 0, 0>);

static_assert(expect_special_layout<layout_map<-1, 2, -1, 1, 0>, 0, 1, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<2, -1, 1, -1, 0>, 1, 0, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<2, 1, -1, 0, -1>, 1, 1, 0, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, 1, 2, -1, 0>, 0, 1, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<2, -1, 1, 0, -1>, 1, 0, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, 1, 2, 0, -1>, 0, 1, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, -1, -1, 1, 0>, 0, 0, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<1, -1, -1, -1, 0>, 1, 0, 0, 0, 1>);
static_assert(expect_special_layout<layout_map<1, 0, -1, -1, -1>, 1, 1, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, 0, 1, -1, -1>, 0, 1, 1, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, 1, 0, -1>, 0, 0, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<0, -1, -1, -1, -1>, 1, 0, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, 0, -1, -1, -1>, 0, 1, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, 0, -1, -1>, 0, 0, 1, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, -1, 0, -1>, 0, 0, 0, 1, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, -1, -1, 0>, 0, 0, 0, 0, 1>);

#elif defined(GT_STORAGE_GPU)

static_assert(expect_layout<0>);
static_assert(expect_layout<1, 0>);
static_assert(expect_layout<2, 1, 0>);
static_assert(expect_layout<3, 2, 1, 0>);
static_assert(expect_layout<4, 3, 2, 1, 0>);

static_assert(expect_special_layout<layout_map<4, 3, 2, 1, 0>, 1, 1, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<-1, 3, 2, 1, 0>, 0, 1, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<3, -1, 2, 1, 0>, 1, 0, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<3, 2, -1, 1, 0>, 1, 1, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<3, 2, 1, -1, 0>, 1, 1, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<3, 2, 1, 0, -1>, 1, 1, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, -1, 2, 1, 0>, 0, 0, 1, 1, 1>);
static_assert(expect_special_layout<layout_map<2, -1, -1, 1, 0>, 1, 0, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<2, 1, -1, -1, 0>, 1, 1, 0, 0, 1>);
static_assert(expect_special_layout<layout_map<2, 1, 0, -1, -1>, 1, 1, 1, 0, 0>);

static_assert(expect_special_layout<layout_map<-1, 2, -1, 1, 0>, 0, 1, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<2, -1, 1, -1, 0>, 1, 0, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<2, 1, -1, 0, -1>, 1, 1, 0, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, 2, 1, -1, 0>, 0, 1, 1, 0, 1>);
static_assert(expect_special_layout<layout_map<2, -1, 1, 0, -1>, 1, 0, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, 2, 1, 0, -1>, 0, 1, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<-1, -1, -1, 1, 0>, 0, 0, 0, 1, 1>);
static_assert(expect_special_layout<layout_map<1, -1, -1, -1, 0>, 1, 0, 0, 0, 1>);
static_assert(expect_special_layout<layout_map<1, 0, -1, -1, -1>, 1, 1, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, 1, 0, -1, -1>, 0, 1, 1, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, 1, 0, -1>, 0, 0, 1, 1, 0>);

static_assert(expect_special_layout<layout_map<0, -1, -1, -1, -1>, 1, 0, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, 0, -1, -1, -1>, 0, 1, 0, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, 0, -1, -1>, 0, 0, 1, 0, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, -1, 0, -1>, 0, 0, 0, 1, 0>);
static_assert(expect_special_layout<layout_map<-1, -1, -1, -1, 0>, 0, 0, 0, 0, 1>);

#endif
