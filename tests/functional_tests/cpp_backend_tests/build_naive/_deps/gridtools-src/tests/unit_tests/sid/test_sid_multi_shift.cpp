/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/multi_shift.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/sid/concept.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        TEST(multi_shift, smoke) {
            double data[15][42];

            auto ptr = sid::get_origin(data)();
            auto strides = sid::get_strides(data);

            sid::multi_shift(ptr, strides, tuple(3_c, 5_c, 2_c));
            EXPECT_EQ(&data[3][5], ptr);

            sid::multi_shift(ptr, strides, tuple(0_c, -2_c));
            EXPECT_EQ(&data[3][3], ptr);

            sid::multi_shift(ptr, strides, tuple(-2));
            EXPECT_EQ(&data[1][3], ptr);
        }
    } // namespace
} // namespace gridtools
