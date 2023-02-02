/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/rename_dimensions.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/simple_ptr_holder.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using sid::property;
        using namespace literals;
        namespace tu = tuple_util;

        struct a {};
        struct b {};
        struct c {};
        struct d {};

        TEST(rename_dimensions, smoke) {
            double data[3][5][7];

            auto src = sid::synthetic()
                           .set<property::origin>(sid::simple_ptr_holder(&data[0][0][0]))
                           .set<property::strides>(hymap::keys<a, b, c>::make_values(5_c * 7_c, 7_c, 1_c))
                           .set<property::upper_bounds>(hymap::keys<a, b>::make_values(3, 5));

            auto testee = sid::rename_dimensions<b, d>(src);
            using testee_t = decltype(testee);

            auto strides = sid::get_strides(testee);
            EXPECT_EQ(35, sid::get_stride<a>(strides));
            EXPECT_EQ(0, sid::get_stride<b>(strides));
            EXPECT_EQ(1, sid::get_stride<c>(strides));
            EXPECT_EQ(7, sid::get_stride<d>(strides));

            static_assert(meta::is_empty<get_keys<sid::lower_bounds_type<testee_t>>>());
            auto u_bound = sid::get_upper_bounds(testee);
            EXPECT_EQ(3, at_key<a>(u_bound));
            EXPECT_EQ(5, at_key<d>(u_bound));
        }

        TEST(rename_dimensions, c_array) {
            double data[3][5][7];

            auto testee = sid::rename_dimensions<decltype(1_c), d>(data);
            using testee_t = decltype(testee);

            auto strides = sid::get_strides(testee);
            EXPECT_EQ(35, (sid::get_stride<integral_constant<int, 0>>(strides)));
            EXPECT_EQ(0, (sid::get_stride<integral_constant<int, 1>>(strides)));
            EXPECT_EQ(1, (sid::get_stride<integral_constant<int, 2>>(strides)));
            EXPECT_EQ(7, sid::get_stride<d>(strides));

            auto l_bound = sid::get_lower_bounds(testee);
            EXPECT_EQ(0, (at_key<integral_constant<int, 0>>(l_bound)));
            EXPECT_EQ(0, at_key<d>(l_bound));

            auto u_bound = sid::get_upper_bounds(testee);
            EXPECT_EQ(3, (at_key<integral_constant<int, 0>>(u_bound)));
            EXPECT_EQ(5, at_key<d>(u_bound));
        }

        TEST(rename_dimensions, rename_twice_and_make_composite) {
            double data[3][5][7];
            auto src = sid::synthetic()
                           .set<property::origin>(sid::host_device::simple_ptr_holder(&data[0][0][0]))
                           .set<property::strides>(hymap::keys<a, b, c>::make_values(5_c * 7_c, 7_c, 1_c))
                           .set<property::upper_bounds>(hymap::keys<a, b>::make_values(3, 5));
            auto testee = sid::rename_dimensions<a, c, b, d>(src);
            static_assert(sid::is_sid<decltype(testee)>::value);
            auto composite = sid::composite::keys<void>::make_values(testee);
            static_assert(sid::is_sid<decltype(composite)>::value);
            sid::get_origin(composite);
        }

        TEST(rename_dimensions, numbered) {
            double data[3][5][7];

            auto testee = sid::rename_numbered_dimensions<a, b, c>(data);
            using testee_t = decltype(testee);

            auto strides = sid::get_strides(testee);
            EXPECT_EQ(35, sid::get_stride<a>(strides));
            EXPECT_EQ(7, sid::get_stride<b>(strides));
            EXPECT_EQ(1, sid::get_stride<c>(strides));

            auto l_bound = sid::get_lower_bounds(testee);
            EXPECT_EQ(0, at_key<a>(l_bound));
            EXPECT_EQ(0, at_key<b>(l_bound));
            EXPECT_EQ(0, at_key<c>(l_bound));

            auto u_bound = sid::get_upper_bounds(testee);
            EXPECT_EQ(3, at_key<a>(u_bound));
            EXPECT_EQ(5, at_key<b>(u_bound));
            EXPECT_EQ(7, at_key<c>(u_bound));
        }
    } // namespace
} // namespace gridtools
