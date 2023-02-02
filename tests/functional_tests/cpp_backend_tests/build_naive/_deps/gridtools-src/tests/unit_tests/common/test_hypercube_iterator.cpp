/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/hypercube_iterator.hpp>

#include <gridtools/common/pair.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace gridtools;

class hypercube_iteration : public ::testing::Test {
    using range_t = std::array<size_t, 2>;
    using data_t = std::vector<std::vector<size_t>>;

  public:
    void expect_ranges(range_t range_i, range_t range_j, range_t range_k) {
        for (size_t i = range_i[0]; i != range_i[1]; ++i)
            for (size_t j = range_j[0]; j != range_j[1]; ++j)
                for (size_t k = range_k[0]; k != range_k[1]; ++k)
                    m_expected.push_back({i, j, k});
    }

    template <class R>
    void call_testee(R &&range) {
        for (auto &&item : make_hypercube_view(range))
            m_actual.push_back({item[0], item[1], item[2]});
    }

    ~hypercube_iteration() { EXPECT_THAT(m_actual, testing::ContainerEq(m_expected)); }

  private:
    data_t m_actual;
    data_t m_expected;
};

TEST_F(hypercube_iteration, view_from_array_of_ranges) {
    expect_ranges({1, 3}, {4, 8}, {2, 10});
    call_testee(array{pair(1, 3), pair(4, 8), pair(2, 10)});
}

TEST_F(hypercube_iteration, view_from_tuple_of_ranges) {
    expect_ranges({1, 3}, {4, 8}, {2, 10});
    call_testee(tuple(pair(1, 3), pair(4, 8), pair(2, 10)));
}

TEST_F(hypercube_iteration, from_array_of_integers) {
    expect_ranges({0, 3}, {0, 8}, {0, 10});
    call_testee(array{3, 8, 10});
}

TEST_F(hypercube_iteration, from_zero_to_zero) {
    expect_ranges({}, {}, {});
    call_testee(array{pair(0, 0), pair(0, 0), pair(0, 0)});
}

TEST_F(hypercube_iteration, from_one_to_one) {
    expect_ranges({}, {}, {});
    call_testee(array{0, 0, 0});
}

TEST(hypercube_view_empty_iteration_space, zero_dimensional_range) {
    int count = 0;
    for (auto it : make_hypercube_view(array<pair<size_t, size_t>, 0>())) {
        static_assert(tuple_util::size<decltype(it)>::value == 0);
        ++count;
    }
    EXPECT_EQ(1, count);
}

TEST(hypercube_view_empty_iteration_space, zero_dimensional_size) {
    int count = 0;
    for (auto it : make_hypercube_view(array<size_t, 0>())) {
        static_assert(tuple_util::size<decltype(it)>::value == 0);
        ++count;
    }
    EXPECT_EQ(1, count);
}
