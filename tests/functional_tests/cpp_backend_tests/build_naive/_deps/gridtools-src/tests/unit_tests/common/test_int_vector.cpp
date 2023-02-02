/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/int_vector.hpp>

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/meta.hpp>

namespace gridtools {
    namespace {
        using namespace literals;
        struct a;
        struct b;
        struct c;

        TEST(plus, integrals) {
            auto m1 = hymap::keys<a, b>::make_values(1, 2l);
            auto m2 = hymap::keys<b, c>::make_values(10, 20u);
            auto m3 = hymap::keys<b>::make_values(100);

            auto testee = int_vector::plus(m1, m2, m3);

            using testee_t = decltype(testee);
            static_assert(std::is_same_v<int, element_at<a, testee_t>>);
            static_assert(std::is_same_v<long int, element_at<b, testee_t>>);
            static_assert(std::is_same_v<unsigned int, element_at<c, testee_t>>);

            EXPECT_EQ(1, at_key<a>(testee));
            EXPECT_EQ(112, at_key<b>(testee));
            EXPECT_EQ(20, at_key<c>(testee));

            using namespace int_vector::arithmetic;
            auto testee2 = m1 + m2;
            EXPECT_EQ(1, at_key<a>(testee2));
            EXPECT_EQ(12, at_key<b>(testee2));
            EXPECT_EQ(20, at_key<c>(testee2));
        }

        TEST(plus, integral_constants) {
            auto m1 = hymap::keys<a, b>::make_values(1_c, 2_c);
            auto m2 = hymap::keys<a, b>::make_values(11_c, 12);

            auto testee = int_vector::plus(m1, m2);

            using testee_t = decltype(testee);
            static_assert(std::is_same_v<integral_constant<int, 12>, element_at<a, testee_t>>);
            static_assert(std::is_same_v<int, element_at<b, testee_t>>);

            EXPECT_EQ(14, at_key<b>(testee));
        }

        TEST(plus, tuple_and_arrays) {
            auto m1 = tuple<int, int>{1, 2};
            auto m2 = array<int, 2>{3, 4};
            auto testee = int_vector::plus(m1, m2);

            EXPECT_EQ(4, (at_key<integral_constant<int, 0>>(testee)));
            EXPECT_EQ(6, (at_key<integral_constant<int, 1>>(testee)));
        }

        TEST(multiply, integrals) {
            auto vec = hymap::keys<a, b>::make_values(1_c, 2);

            auto testee = int_vector::multiply(vec, 2);

            EXPECT_EQ(2, at_key<a>(testee));
            EXPECT_EQ(4, at_key<b>(testee));

            using namespace int_vector::arithmetic;
            auto testee2 = vec * 2;
            EXPECT_EQ(2, at_key<a>(testee2));
            EXPECT_EQ(4, at_key<b>(testee2));

            auto testee3 = 2 * vec;
            EXPECT_EQ(2, at_key<a>(testee3));
            EXPECT_EQ(4, at_key<b>(testee3));
        }

        TEST(multiply, integral_constants) {
            auto vec = hymap::keys<a, b>::make_values(1_c, 2);

            auto testee = int_vector::multiply(vec, 2_c);

            using testee_t = decltype(testee);
            static_assert(element_at<a, testee_t>::value == 2);
            EXPECT_EQ(4, at_key<b>(testee));
        }

        TEST(prune_zeros, smoke) {
            auto vec = hymap::keys<a, b, c>::make_values(1, 0_c, 2_c);

            auto testee = int_vector::prune_zeros(vec);

            EXPECT_EQ(1, at_key<a>(testee));
            using testee_t = decltype(testee);
            static_assert(!has_key<testee_t, b>());
            static_assert(element_at<c, testee_t>::value == 2);
        }

        TEST(unary_ops, smoke) {
            using namespace int_vector::arithmetic;

            auto vec = hymap::keys<a, b, c>::make_values(1, 0_c, 2_c);

            auto testee = -vec;

            EXPECT_EQ(-1, at_key<a>(testee));
            using testee_t = decltype(testee);
            static_assert(element_at<b, testee_t>::value == 0);
            static_assert(element_at<c, testee_t>::value == -2);

            auto testee2 = +vec;
            EXPECT_EQ(1, at_key<a>(testee2));
            using testee2_t = decltype(testee2);
            static_assert(element_at<b, testee2_t>::value == 0);
            static_assert(element_at<c, testee2_t>::value == 2);
        }

        TEST(minus_op, smoke) {
            using namespace int_vector::arithmetic;

            auto m1 = hymap::keys<a, b>::make_values(1, 2_c);
            auto m2 = hymap::keys<a, b, c>::make_values(1, 1_c, 3);

            auto testee = m1 - m2;

            EXPECT_EQ(0, at_key<a>(testee));
            using testee_t = decltype(testee);
            static_assert(element_at<b, testee_t>::value == 1);
            EXPECT_EQ(-3, at_key<c>(testee));
        }
    } // namespace
} // namespace gridtools
