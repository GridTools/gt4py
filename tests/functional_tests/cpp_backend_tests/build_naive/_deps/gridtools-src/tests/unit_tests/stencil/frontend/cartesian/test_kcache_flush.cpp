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

    using axis_t = axis<3, axis_config::offset_limit<3>>;
    using kfull = axis_t::full_interval;

    double in(int i, int j, int k) { return i + j + k + 1; }

    GT_ENVIRONMENT_TEST_SUITE(test_kcache_flush,
        (vertical_test_environment<0, axis_t>),
        stencil_backend_t,
        (double, inlined_params<6, 6, 2, 6, 2>));

    struct shift_acc_forward_flush {
        using in = in_accessor<0>;
        using out = inout_accessor<1, extent<0, 0, 0, 0, -1, 0>>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
            eval(out()) = eval(in());
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, 0>) {
            eval(out()) = eval(out(0, 0, -1)) + eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_flush, forward) {
        auto out = TypeParam::make_storage();
        auto spec = [](auto in, auto out) {
            return execute_forward().k_cached(cache_io_policy::flush(), out).stage(shift_acc_forward_flush(), in, out);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), TypeParam::make_storage(in), out);
        auto ref = TypeParam::make_storage();
        auto refv = ref->host_view();
        for (int i = 0; i < TypeParam::d(0); ++i)
            for (int j = 0; j < TypeParam::d(1); ++j) {
                refv(i, j, 0) = in(i, j, 0);
                for (int k = 1; k < 10; ++k)
                    refv(i, j, k) = refv(i, j, k - 1) + in(i, j, k);
            }
        TypeParam::verify(ref, out);
    }

    struct shift_acc_backward_flush {
        using in = in_accessor<0>;
        using out = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
            eval(out()) = eval(in());
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::modify<0, -1>) {
            eval(out()) = eval(out(0, 0, 1)) + eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_flush, backward) {
        auto out = TypeParam::make_storage();
        auto spec = [](auto in, auto out) {
            return execute_backward()
                .k_cached(cache_io_policy::flush(), out)
                .stage(shift_acc_backward_flush(), in, out);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), TypeParam::make_storage(in), out);
        auto ref = TypeParam::make_storage();
        auto refv = ref->host_view();
        for (int i = 0; i < TypeParam::d(0); ++i)
            for (int j = 0; j < TypeParam::d(1); ++j) {
                refv(i, j, 9) = in(i, j, 9);
                for (int k = 8; k >= 0; --k)
                    refv(i, j, k) = refv(i, j, k + 1) + in(i, j, k);
            }
        TypeParam::verify(ref, out);
    }
} // namespace
