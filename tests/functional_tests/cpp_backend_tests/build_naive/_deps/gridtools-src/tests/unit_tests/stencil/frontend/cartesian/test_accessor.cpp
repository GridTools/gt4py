/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil/frontend/cartesian/accessor.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil/frontend/cartesian/expressions.hpp>

using namespace gridtools;
using namespace stencil;
using namespace cartesian;
using namespace expressions;

static_assert(is_accessor<inout_accessor<6, extent<3, 4, 4, 5>>>::value);
static_assert(is_accessor<in_accessor<2>>::value);
static_assert(!is_accessor<int>::value);
static_assert(!is_accessor<double &>::value);
static_assert(!is_accessor<double const &>::value);

TEST(accessor, smoke) {
    using testee_t = inout_accessor<0, extent<0, 3, 0, 2, -1, 0>>;
    static_assert(tuple_util::size<testee_t>::value == 3);

    testee_t testee{3, 2, -1};

    EXPECT_EQ(3, tuple_util::get<0>(testee));
    EXPECT_EQ(2, tuple_util::get<1>(testee));
    EXPECT_EQ(-1, tuple_util::get<2>(testee));
}

TEST(accessor, zero_accessor) {
    using testee_t = accessor<0>;
    static_assert(tuple_util::size<testee_t>::value == 0);
    testee_t{0, 0, 0, 0};
    testee_t{dimension<3>{}};
}

TEST(accessor, extra_args) {
    using testee_t = inout_accessor<0, extent<-1, 1>>;
    static_assert(tuple_util::size<testee_t>::value == 1);
    testee_t{1, 0};
    testee_t{dimension<2>{0}};
}

/**
 * @brief interface with out-of-order optional arguments
 */
TEST(accessor, alternative1) {
    inout_accessor<0, extent<>, 6> first(dimension<6>(-6), dimension<4>(12));

    EXPECT_EQ(0, tuple_util::get<0>(first));
    EXPECT_EQ(0, tuple_util::get<1>(first));
    EXPECT_EQ(0, tuple_util::get<2>(first));
    EXPECT_EQ(12, tuple_util::get<3>(first));
    EXPECT_EQ(0, tuple_util::get<4>(first));
    EXPECT_EQ(-6, tuple_util::get<5>(first));
}

/**
 * @brief interface with out-of-order optional arguments, represented as matlab indices
 */
TEST(accessor, alternative2) {
    constexpr dimension<1> i;
    constexpr dimension<2> j;

    constexpr dimension<4> t;
    inout_accessor<0, extent<-5, 0, 0, 0, 0, 8>, 4> first(i - 5, j, dimension<3>(8), t + 2);

    EXPECT_EQ(-5, tuple_util::get<0>(first));
    EXPECT_EQ(0, tuple_util::get<1>(first));
    EXPECT_EQ(8, tuple_util::get<2>(first));
    EXPECT_EQ(2, tuple_util::get<3>(first));
}
