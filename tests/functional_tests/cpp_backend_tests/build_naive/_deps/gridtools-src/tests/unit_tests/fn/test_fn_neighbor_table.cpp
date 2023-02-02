/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/neighbor_table.hpp>

#include <array>
#include <tuple>

#include <gtest/gtest.h>

namespace gridtools::fn {
    struct a_neighbor_table {};
    std::array<int, 3> neighbor_table_neighbors(a_neighbor_table, int) { return {1, 2, 42}; }

    static_assert(neighbor_table::is_neighbor_table<a_neighbor_table>());
    static_assert(neighbor_table::is_neighbor_table<std::array<int, 3>[]>());
    static_assert(neighbor_table::is_neighbor_table<std::tuple<int, int, int>[]>());
    static_assert(!neighbor_table::is_neighbor_table<int>());

    TEST(neighbor_table, smoke) {
        std::array<int, 2> table[3] = {{1, 2}, {3, 4}, {4, 5}};
        for (int i = 0; i < 3; ++i)
            EXPECT_EQ(table[i], neighbor_table::neighbors(table, i));
    }
} // namespace gridtools::fn
