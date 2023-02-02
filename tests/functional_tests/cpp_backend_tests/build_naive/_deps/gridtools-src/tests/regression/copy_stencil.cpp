/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct copy_functor {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(in());
        }
    };

    GT_REGRESSION_TEST(copy_stencil, test_environment<>, stencil_backend_t) {
        auto in = [](int i, int j, int k) { return i + j + k; };
        auto out = TypeParam::make_storage();
        auto comp = [&out, grid = TypeParam::make_grid(), in = TypeParam::make_const_storage(in)] {
            run_single_stage(copy_functor(), stencil_backend_t(), grid, in, out);
        };
        comp();
        TypeParam::verify(in, out);
        TypeParam::benchmark("copy_stencil", comp);
    }
} // namespace
