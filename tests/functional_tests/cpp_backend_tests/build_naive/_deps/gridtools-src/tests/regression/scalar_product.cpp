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

#include <cstdlib>

#include <gridtools/reduction.hpp>
#include <gridtools/stencil/cartesian.hpp>

#include <reduction_select.hpp>
#include <test_environment.hpp>
#include <verifier.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct mul_functor {
        using out = inout_accessor<0>;
        using lhs = in_accessor<1>;
        using rhs = in_accessor<2>;

        using param_list = make_param_list<out, lhs, rhs>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(lhs()) * eval(rhs());
        }
    };

    GT_REGRESSION_TEST(scalar_product, test_environment<>, reduction_backend_t) {
        using float_t = typename TypeParam::float_t;
        auto init = [](int, int, int) { return std::rand(); };
        auto comp = [out = reduction::make_reducible<reduction_backend_t, storage_traits_t>(
                         float_t(0), TypeParam::d(0), TypeParam::d(1), TypeParam::d(2)),
                        grid = TypeParam::make_grid(),
                        lhs = TypeParam::make_const_storage(init),
                        rhs = TypeParam::make_const_storage(init)] {
            run_single_stage(mul_functor(), stencil_backend_t(), grid, out, lhs, rhs);
            return out.reduce(reduction::plus());
        };
        TypeParam::benchmark("scalar_product", comp);
    }

    struct fill_functor {
        using out = inout_accessor<0>;
        using param_list = make_param_list<out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = 1;
        }
    };

    GT_REGRESSION_TEST(summation, test_environment<>, reduction_backend_t) {
        using float_t = typename TypeParam::float_t;
        auto comp = [out = reduction::make_reducible<reduction_backend_t, storage_traits_t>(
                         float_t(0), TypeParam::d(0), TypeParam::d(1), TypeParam::d(2)),
                        grid = TypeParam::make_grid()] {
            run_single_stage(fill_functor(), stencil_backend_t(), grid, out);
            return out.reduce(reduction::plus());
        };
        EXPECT_NEAR(comp(), TypeParam::d(0) * TypeParam::d(1) * TypeParam::d(2), default_precision<float_t>());
    }

} // namespace
