/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/frontend/axis.hpp>

#include <type_traits>

#include <gtest/gtest.h>

using namespace gridtools;
using namespace stencil;
using namespace core;

constexpr int level_offset_limit = 2;

template <uint_t Splitter, int_t Offset>
using level_t = level<Splitter, Offset, level_offset_limit>;

TEST(test_axis, ctor) {
    using axis_t = axis<2, axis_config::offset_limit<level_offset_limit>>;
    auto axis_ = axis_t((uint_t)5, (uint_t)4);

    EXPECT_EQ(5, axis_.interval_size(0));
    EXPECT_EQ(4, axis_.interval_size(1));
}

namespace intervals {
    using axis_t = axis<3, axis_config::offset_limit<level_offset_limit>>;

    template <class T, uint_t FromSplitter, int_t FromOffset, uint_t ToSplitter, int_t ToOffset>
    constexpr bool testee =
        std::is_same_v<T, interval<level_t<FromSplitter, FromOffset>, level_t<ToSplitter, ToOffset>>>;

    // full interval
    static_assert(testee<axis_t::full_interval, 0, 1, 3, -1>);

    // intervals by id
    static_assert(testee<axis_t::get_interval<0>, 0, 1, 1, -1>);
    static_assert(testee<axis_t::get_interval<1>, 1, 1, 2, -1>);

    // hull of multiple intervals
    static_assert(testee<axis_t::get_interval<1, 2>, 1, 1, 3, -1>);
} // namespace intervals
