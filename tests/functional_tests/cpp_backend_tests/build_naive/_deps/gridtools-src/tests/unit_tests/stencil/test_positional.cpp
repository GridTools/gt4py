/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/positional.hpp>

#include <gtest/gtest.h>

#include <gridtools/sid/concept.hpp>

namespace gridtools {
    namespace stencil {
        namespace {
            struct d;
            using testee_t = positional<d>;

            static_assert(is_sid<testee_t>());

            TEST(positional, smoke) {
                testee_t testee{1};

                auto ptr = sid::get_origin(testee)();

                EXPECT_EQ(*ptr, 1);

                auto strides = sid::get_strides(testee);

                sid::shift(ptr, sid::get_stride<d>(strides), -34);

                EXPECT_EQ(*ptr, -33);

                using diff_t = sid::ptr_diff_type<testee_t>;

                diff_t diff{};

                sid::shift(diff, sid::get_stride<d>(strides), -34);

                ptr = sid::get_origin(testee)() + diff;

                EXPECT_EQ(*ptr, -33);
            }
        } // namespace
    }     // namespace stencil
} // namespace gridtools
