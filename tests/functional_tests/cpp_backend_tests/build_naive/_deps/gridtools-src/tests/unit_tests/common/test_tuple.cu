/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/tuple.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools {
    TEST(tuple, get) {
        tuple<int, double> src{42, 2.5};
        EXPECT_EQ(42, on_device::exec(tuple_util::host_device::get_nth_f<0>{}, src));
        EXPECT_EQ(2.5, on_device::exec(tuple_util::host_device::get_nth_f<1>{}, src));
    }

    __device__ tuple<int, double> element_wise_ctor(int x, double y) { return {x, y}; }

    TEST(tuple, element_wise_ctor) {
        tuple<int, double> testee = on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&element_wise_ctor), 42, 2.5);
        EXPECT_EQ(42, tuple_util::host::get<0>(testee));
        EXPECT_EQ(2.5, tuple_util::host::get<1>(testee));
    }

#if defined(__clang__) && (__clang_major__ == 11 && __clang_minor__ == 0 && __clang_patchlevel__ == 0)
    // crashes Clang 11.0.0, see https://github.com/GridTools/gridtools/issues/1615
#else
    __device__ tuple<int, double> element_wise_conversion_ctor(char x, char y) { return {x, y}; }

    TEST(tuple, element_wise_conversion_ctor) {
        tuple<int, double> testee =
            on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&element_wise_conversion_ctor), 'a', 'b');
        EXPECT_EQ('a', tuple_util::host::get<0>(testee));
        EXPECT_EQ('b', tuple_util::host::get<1>(testee));
    }

    __device__ tuple<int, double> tuple_conversion_ctor(tuple<char, char> const &src) { return src; }

    TEST(tuple, tuple_conversion_ctor) {
        tuple<int, double> testee =
            on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&tuple_conversion_ctor), tuple<char, char>{'a', 'b'});
        EXPECT_EQ('a', tuple_util::host::get<0>(testee));
        EXPECT_EQ('b', tuple_util::host::get<1>(testee));
    }
#endif
} // namespace gridtools

#include "test_tuple.cpp"
