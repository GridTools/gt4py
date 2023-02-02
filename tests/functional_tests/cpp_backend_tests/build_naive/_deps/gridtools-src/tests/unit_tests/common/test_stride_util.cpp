/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/stride_util.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>

namespace gridtools {
    namespace stride_util {
        namespace {
            using namespace literals;
            namespace tu = tuple_util;

            TEST(make_strides_from_sizes, smoke) {
                auto testee = make_strides_from_sizes(tuple(20, 30, 40));
                static_assert(std::is_same_v<decltype(testee), tuple<integral_constant<int, 1>, int, int>>);

                EXPECT_EQ(1, tu::get<0>(testee));
                EXPECT_EQ(20, tu::get<1>(testee));
                EXPECT_EQ(600, tu::get<2>(testee));
            }

            TEST(make_strides_from_sizes, integral_constants) {
                auto testee = make_strides_from_sizes(tuple(20_c, 30_c, 40));
                static_assert(std::is_same_v<decltype(testee),
                    tuple<integral_constant<int, 1>, integral_constant<int, 20>, integral_constant<int, 600>>>);

                EXPECT_EQ(1, tu::get<0>(testee));
                EXPECT_EQ(20, tu::get<1>(testee));
                EXPECT_EQ(600, tu::get<2>(testee));
            }

            struct a;
            struct b;
            struct c;

            TEST(make_strides_from_sizes, hymap) {
                auto testee = make_strides_from_sizes(hymap::keys<a, b, c>::make_values(20, 30, 40));
                static_assert(
                    std::is_same_v<decltype(testee), hymap::keys<a, b, c>::values<integral_constant<int, 1>, int, int>>);

                EXPECT_EQ(1, at_key<a>(testee));
                EXPECT_EQ(20, at_key<b>(testee));
                EXPECT_EQ(600, at_key<c>(testee));
            }

            TEST(total_size, smoke) {
                auto testee = total_size(tuple(20, 30, 40));
                EXPECT_EQ(24000, testee);
            }

            TEST(total_size, integral_constants) {
                auto testee = total_size(tuple(20_c, 30_c, 40_c));
                static_assert(std::is_same_v<decltype(testee), integral_constant<int, 24000>>);

                EXPECT_EQ(24000, testee);
            }
        } // namespace
    }     // namespace stride_util
} // namespace gridtools
