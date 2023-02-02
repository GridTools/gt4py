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

    struct parallel_functor {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;
        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, axis<2>::get_interval<0>) {
            eval(out()) = eval(in());
        }
        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, axis<2>::get_interval<1>) {
            eval(out()) = 2 * eval(in());
        }
    };

    using env1_t = test_environment<0, axis<2>>::apply<stencil_backend_t, double, inlined_params<7, 8, 14, 16>>;

    using kparallel_with_temporary = regression_test<env1_t>;

    TEST_F(kparallel_with_temporary, test) {
        auto in = [](int i, int j, int k) { return i * 1000 + j * 100 + k; };
        auto out = env1_t::make_storage(1.5);
        run(
            [](auto in, auto out) {
                GT_DECLARE_TMP(double, tmp);
                return execute_parallel().stage(parallel_functor(), in, tmp).stage(parallel_functor(), tmp, out);
            },
            stencil_backend_t(),
            env1_t::make_grid(),
            env1_t ::make_storage(in),
            out);
        env1_t::verify([&](int i, int j, int k) { return in(i, j, k) * (k < 14 ? 1 : 4); }, out);
    }

    struct parallel_functor_on_upper_interval {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;
        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, axis<3>::get_interval<1>) {
            eval(out()) = eval(in());
        }
    };

    using env2_t = test_environment<0, axis<3>>::apply<stencil_backend_t, double, inlined_params<7, 8, 14, 16, 18>>;

    using kparallel_with_unused_intervals = regression_test<env2_t>;

    TEST_F(kparallel_with_unused_intervals, test) {
        auto in = [](int i, int j, int k) { return i * 1000 + j * 100 + k; };
        auto out = env2_t::make_storage(1.5);
        run_single_stage(parallel_functor_on_upper_interval(),
            stencil_backend_t(),
            env2_t::make_grid(),
            env2_t::make_storage(in),
            out);
        env2_t::verify([&in](int i, int j, int k) { return k >= 14 && k < 14 + 16 ? in(i, j, k) : 1.5; }, out);
    }
} // namespace
