/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/contiguous.hpp>

#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/concept.hpp>

namespace gridtools {
    namespace {

        namespace tu = tuple_util;
        using namespace literals;

        struct a;
        struct b;
        struct c;

        using sid::property;

        TEST(contiguous, smoke) {
            auto alloc = sid::allocator(&std::make_unique<char[]>);
            hymap::keys<a, b, c>::values<integral_constant<int_t, 2>, int, int> sizes = {2_c, 3, 4};
            auto testee = sid::make_contiguous<int>(alloc, sizes);

            using testee_t = decltype(testee);
            using ptr_diff_t = sid::ptr_diff_type<testee_t>;
            using strides_t = sid::strides_type<testee_t>;
            using strides_kind_t = sid::strides_kind<testee_t>;

            static_assert(is_sid<testee_t>());
            static_assert(std::is_same_v<ptrdiff_t, ptr_diff_t>);
            static_assert(std::is_same_v<strides_kind_t, strides_t>);

            auto lower_bounds = sid::get_lower_bounds(testee);
            EXPECT_EQ(0, at_key<a>(lower_bounds));
            EXPECT_EQ(0, at_key<b>(lower_bounds));
            EXPECT_EQ(0, at_key<c>(lower_bounds));

            auto upper_bounds = sid::get_upper_bounds(testee);
            EXPECT_EQ(2, at_key<a>(upper_bounds));
            EXPECT_EQ(3, at_key<b>(upper_bounds));
            EXPECT_EQ(4, at_key<c>(upper_bounds));

            auto strides = sid::get_strides(testee);
            auto origin = sid::get_origin(testee)();

            *origin = 42;
            EXPECT_EQ(42, *origin);

            auto ptr = origin;
            sid::shift(ptr, sid::get_stride<a>(strides), 1);
            sid::shift(ptr, sid::get_stride<b>(strides), 2);
            sid::shift(ptr, sid::get_stride<c>(strides), 3);

            *ptr = 88;
            EXPECT_EQ(88, *ptr);
        }
    } // namespace
} // namespace gridtools
