/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/layout_transformation.hpp>

#include <storage_select.hpp>
#include <test_environment.hpp>

using namespace gridtools;

template <typename Src, typename Dst>
void verify_result(Src &src, Dst &dst) {
    auto src_v = src->const_host_view();
    auto dst_v = dst->const_host_view();

    auto &&lengths = src->lengths();
    for (int i = 0; i < lengths[0]; ++i)
        for (int j = 0; j < lengths[1]; ++j)
            for (int k = 0; k < lengths[2]; ++k)
                EXPECT_EQ(src_v(i, j, k), dst_v(i, j, k));
}

GT_REGRESSION_TEST(layout_transformation, test_environment<>, storage_traits_t) {
    auto src =
        TypeParam::builder().template layout<0, 1, 2>().initializer([](int i, int j, int k) { return i + j + k; })();
    auto dst = TypeParam::builder().template layout<2, 1, 0>()();
    auto testee = [&] {
        transform_layout(dst->get_target_ptr(), src->get_target_ptr(), src->lengths(), dst->strides(), src->strides());
    };
    testee();
    verify_result(src, dst);
    TypeParam::benchmark("layout_transformation", testee);
}
