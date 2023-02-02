/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/frontend/make_grid.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil/core/interval.hpp>
#include <gridtools/stencil/frontend/axis.hpp>

using namespace gridtools;
using namespace stencil;

TEST(test_grid, k_total_length) {
    using axis_t = axis<1, axis_config::offset_limit<3>>;
    auto testee = make_grid(1, 1, axis_t{45});

    EXPECT_EQ(45, testee.k_size());
}

constexpr int level_offset_limit = 3;

template <size_t N>
using axis_type = axis<N, axis_config::offset_limit<level_offset_limit>>;

template <uint_t Splitter, int_t Offset>
using level_type = core::level<Splitter, Offset, level_offset_limit>;

TEST(test_grid, make_grid_makes_splitters_and_values) {
    using axis_t = axis<2, axis_config::offset_limit<level_offset_limit>>;

    using interval1_t = core::interval<level_type<0, 1>, level_type<1, -1>>;
    using interval2_t = core::interval<level_type<1, 1>, level_type<2, -1>>;

    auto testee = make_grid(1, 1, axis_type<2>{5, 10});

    EXPECT_EQ(0, testee.k_start());
    EXPECT_EQ(15, testee.k_size());

    EXPECT_EQ(0, testee.k_start(interval1_t()));
    EXPECT_EQ(5, testee.k_size(interval1_t()));

    EXPECT_EQ(5, testee.k_start(interval2_t()));
    EXPECT_EQ(10, testee.k_size(interval2_t()));
}
