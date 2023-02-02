/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/pair.hpp>

#include <type_traits>

#include <gtest/gtest.h>

TEST(pair, non_uniform_ctor) {
    int int_val = 1;
    size_t size_t_val = 2;

    gridtools::pair<size_t, size_t> my_pair(int_val, size_t_val);

    EXPECT_EQ((size_t)int_val, my_pair.first);
    EXPECT_EQ(size_t_val, my_pair.second);
}

TEST(pair, get_rval_ref) {
    size_t val0 = 1;
    size_t val1 = 2;

    EXPECT_TRUE(
        std::is_rvalue_reference<decltype(gridtools::get<0>(gridtools::pair<size_t, size_t>{val0, val1}))>::value);
    EXPECT_EQ(val0, gridtools::get<0>(gridtools::pair<size_t, size_t>{val0, val1}));
    EXPECT_EQ(val1, gridtools::get<1>(gridtools::pair<size_t, size_t>{val0, val1}));
}

TEST(pair, eq) {
    gridtools::pair<size_t, size_t> pair1{1, 2};
    gridtools::pair<size_t, size_t> pair2{pair1};

    EXPECT_TRUE(pair1 == pair2);
    EXPECT_FALSE(pair1 != pair2);
    EXPECT_FALSE(pair1 < pair2);
    EXPECT_TRUE(pair1 <= pair2);
    EXPECT_FALSE(pair1 > pair2);
    EXPECT_TRUE(pair1 >= pair2);
}
TEST(pair, compare_first_differ) {
    gridtools::pair<size_t, size_t> smaller{1, 2};
    gridtools::pair<size_t, size_t> bigger{2, 2};

    EXPECT_FALSE(smaller == bigger);
    EXPECT_TRUE(smaller != bigger);
    EXPECT_TRUE(smaller < bigger);
    EXPECT_TRUE(smaller <= bigger);
    EXPECT_FALSE(smaller > bigger);
    EXPECT_FALSE(smaller >= bigger);
}

TEST(pair, lt_gt_second_differ) {
    gridtools::pair<size_t, size_t> smaller{1, 1};
    gridtools::pair<size_t, size_t> bigger{1, 2};

    EXPECT_FALSE(smaller == bigger);
    EXPECT_TRUE(smaller != bigger);
    EXPECT_TRUE(smaller < bigger);
    EXPECT_TRUE(smaller <= bigger);
    EXPECT_FALSE(smaller > bigger);
    EXPECT_FALSE(smaller >= bigger);
}

TEST(pair, construct_from_std_pair) {
    std::pair<size_t, size_t> std_pair{1, 2};

    gridtools::pair<size_t, size_t> gt_pair(std_pair);

    ASSERT_EQ(std_pair.first, gt_pair.first);
    ASSERT_EQ(std_pair.second, gt_pair.second);
}
