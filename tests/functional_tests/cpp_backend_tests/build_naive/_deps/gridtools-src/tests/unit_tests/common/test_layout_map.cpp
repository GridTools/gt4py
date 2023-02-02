/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/layout_map.hpp>

using namespace gridtools;

namespace simple_layout {
    typedef layout_map<0, 1, 2> layout1;

    // test length
    static_assert(layout1::masked_length == 3);
    static_assert(layout1::unmasked_length == 3);

    // test find method
    static_assert(layout1::find(0) == 0);
    static_assert(layout1::find(1) == 1);
    static_assert(layout1::find(2) == 2);

    // test at method
    static_assert(layout1::at(0) == 0);
    static_assert(layout1::at(1) == 1);
    static_assert(layout1::at(2) == 2);
} // namespace simple_layout

namespace extended_layout {
    typedef layout_map<3, 2, 1, 0> layout2;

    // test length
    static_assert(layout2::masked_length == 4);
    static_assert(layout2::unmasked_length == 4);

    // test find method
    static_assert(layout2::find(0) == 3);
    static_assert(layout2::find(1) == 2);
    static_assert(layout2::find(2) == 1);
    static_assert(layout2::find(3) == 0);

    // test at method
    static_assert(layout2::at(0) == 3);
    static_assert(layout2::at(1) == 2);
    static_assert(layout2::at(2) == 1);
    static_assert(layout2::at(3) == 0);
} // namespace extended_layout

namespace masked_layout {
    typedef layout_map<2, -1, 1, 0> layout3;

    // test length
    static_assert(layout3::masked_length == 4);
    static_assert(layout3::unmasked_length == 3);

    // test find method
    static_assert(layout3::find(0) == 3);
    static_assert(layout3::find(1) == 2);
    static_assert(layout3::find(2) == 0);

    // test at method
    static_assert(layout3::at(0) == 2);
    static_assert(layout3::at(1) == -1);
    static_assert(layout3::at(2) == 1);
    static_assert(layout3::at(3) == 0);
} // namespace masked_layout
