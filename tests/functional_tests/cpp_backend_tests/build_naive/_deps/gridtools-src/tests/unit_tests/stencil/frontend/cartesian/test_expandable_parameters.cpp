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
#include <gridtools/stencil/naive.hpp>

#define GT_STENCIL_NAIVE
#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;
    using namespace expressions;

    struct expandable_parameters : ::testing::Test {
        using env_t = test_environment<>::apply<naive, double, inlined_params<13, 9, 7>>;
        using storages_t = std::vector<env_t::storage_type>;

        template <class Comp, class... Args>
        void run_computation(Comp comp, Args &&...args) const {
            expandable_run<2>(comp, naive(), env_t::make_grid(), std::forward<Args>(args)...);
        }

        void verify(storages_t const &expected, storages_t const &actual) const {
            EXPECT_EQ(expected.size(), actual.size());
            for (size_t i = 0; i != expected.size(); ++i)
                env_t::verify(expected[i], actual[i]);
        }
    };

    struct expandable_parameters_copy : expandable_parameters {

        storages_t out = {env_t::make_storage(1.),
            env_t::make_storage(2.),
            env_t::make_storage(3.),
            env_t::make_storage(4.),
            env_t::make_storage(5.)};
        storages_t in = {env_t::make_storage(-1.),
            env_t::make_storage(-2.),
            env_t::make_storage(-3.),
            env_t::make_storage(-4.),
            env_t::make_storage(-5.)};

        template <class Functor>
        void run_computation() {
            expandable_parameters::run_computation(
                [](auto out, auto in) { return execute_parallel().stage(Functor(), out, in); }, out, in);
        }

        ~expandable_parameters_copy() { verify(in, out); }
    };

    struct copy_functor {
        typedef inout_accessor<0> out;
        typedef in_accessor<1> in;

        typedef make_param_list<out, in> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out{}) = eval(in{});
        }
    };

    TEST_F(expandable_parameters_copy, copy) { run_computation<copy_functor>(); }

    struct copy_functor_with_expression {
        typedef inout_accessor<0> out;
        typedef in_accessor<1> in;

        typedef make_param_list<out, in> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            // use an expression which is equivalent to a copy to simplify the check
            eval(out{}) = eval(2. * in{} - in{});
        }
    };

    TEST_F(expandable_parameters_copy, copy_with_expression) { run_computation<copy_functor_with_expression>(); }

    struct call_proc_copy_functor {
        typedef inout_accessor<0> out;
        typedef in_accessor<1> in;

        typedef make_param_list<out, in> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            call_proc<copy_functor>::with(eval, out(), in());
        }
    };

    TEST_F(expandable_parameters_copy, call_proc_copy) { run_computation<call_proc_copy_functor>(); }

    struct call_copy_functor {
        typedef inout_accessor<0> out;
        typedef in_accessor<1> in;

        typedef make_param_list<out, in> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = call<copy_functor>::with(eval, in());
        }
    };

    TEST_F(expandable_parameters_copy, call_copy) { run_computation<call_copy_functor>(); }

    struct shift_functor {
        typedef inout_accessor<0, extent<0, 0, 0, 0, -1, 0>> out;

        typedef make_param_list<out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(out(0, 0, -1));
        }
    };

    struct call_shift_functor {
        typedef inout_accessor<0, extent<0, 0, 0, 0, -1, 0>> out;

        typedef make_param_list<out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, axis<1>::full_interval::modify<1, 0>) {
            call_proc<shift_functor>::with(eval, out());
        }
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &, axis<1>::full_interval::first_level) {}
    };

    TEST_F(expandable_parameters, call_shift) {
        auto expected = [&](double value) { return env_t::make_storage([=](int_t, int_t, int_t) { return value; }); };
        auto in = [&](double value) {
            return env_t::make_storage([=](int_t, int_t, int_t k) { return k == 0 ? value : -1; });
        };

        storages_t actual = {in(14), in(15), in(16), in(17), in(18)};
        run_computation([](auto x) { return execute_forward().stage(call_shift_functor(), x); }, actual);
        verify({expected(14), expected(15), expected(16), expected(17), expected(18)}, actual);
    }

    TEST_F(expandable_parameters, caches) {
        storages_t out = {env_t::make_storage(1.),
            env_t::make_storage(2.),
            env_t::make_storage(3.),
            env_t::make_storage(4.),
            env_t::make_storage(5.)};
        auto in = env_t::make_storage(42.);
        run_computation(
            [](auto in, auto out) {
                GT_DECLARE_TMP(double, tmp);
                return execute_parallel().ij_cached(tmp).stage(copy_functor(), tmp, in).stage(copy_functor(), out, tmp);
            },
            in,
            out);
        verify({in, in, in, in, in}, out);
    }
} // namespace
