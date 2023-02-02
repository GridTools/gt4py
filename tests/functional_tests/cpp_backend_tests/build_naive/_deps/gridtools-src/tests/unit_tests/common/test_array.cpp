/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/array.hpp>
#include <gridtools/common/array_addons.hpp>

#include <gtest/gtest.h>

using namespace gridtools;

TEST(array, test_copyctr) {
    array<uint_t, 4> a{4, 2, 3, 1};
    auto mod_a(a);
    EXPECT_EQ(mod_a, a);
    EXPECT_EQ(mod_a[0], 4);
}

TEST(array, iterate_empty) {
    array<uint_t, 0> a;
    EXPECT_EQ(a.begin(), a.end());
    for (auto &&el : a) {
        (void)el;
        FAIL();
    }
}

namespace constexpr_compare {
    constexpr array<uint_t, 3> a{0, 0, 0};
    constexpr array<uint_t, 3> b{0, 0, 0};
    constexpr array<uint_t, 3> c{0, 0, 1};

    static_assert(a == b);
    static_assert(a != c);
} // namespace constexpr_compare

TEST(array, iterate) {
    const int N = 5;
    array<double, N> a{};

    EXPECT_EQ(N, std::distance(a.begin(), a.end()));

    int count = 0;
    for (auto &&el : a) {
        (void)el;
        count++;
    }

    EXPECT_EQ(N, count);
}
