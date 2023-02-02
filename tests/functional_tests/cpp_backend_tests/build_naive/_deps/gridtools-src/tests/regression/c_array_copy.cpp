/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/naive.hpp>

using namespace gridtools;
using namespace stencil;
using namespace cartesian;

struct copy {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using param_list = make_param_list<in, out>;
    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in());
    }
};

template <class T>
auto grid(T const &) {
    static_assert(std::rank<T>() == 3);
    return make_grid(std::extent<T, 0>(), std::extent<T, 1>(), std::extent<T, 2>());
}

TEST(c_array, smoke) {
    int out[3][4][5];
    decltype(out) in;
    for (auto &&vvv : in)
        for (auto &&vv : vvv)
            for (auto &&v : vv)
                v = 42;
    run_single_stage(copy(), naive(), grid(out), in, out);
    EXPECT_THAT(out, testing::ContainerEq(in));
}
