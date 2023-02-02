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

    GT_ENVIRONMENT_TEST_SUITE(test_kcache_fill_and_flush,
        (vertical_test_environment<0, axis_t>),
        stencil_backend_t,
        (double, inlined_params<6, 6, 2, 6, 2>));

    struct shift_acc_forward_fill_and_flush {
        using in = inout_accessor<0, extent<0, 0, 0, 0, -1, 0>>;
        using param_list = make_param_list<in>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, 0>) {
            eval(in()) = eval(in()) + eval(in(0, 0, -1));
        }
        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
            eval(in()) = eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_fill_and_flush, forward) {
        auto field = TypeParam::make_storage(in);
        auto spec = [](auto in) {
            return execute_forward()
                .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in)
                .stage(shift_acc_forward_fill_and_flush(), in);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), field);
        auto ref = TypeParam::make_storage();
        auto refv = ref->host_view();
        for (int i = 0; i < TypeParam::d(0); ++i)
            for (int j = 0; j < TypeParam::d(1); ++j) {
                refv(i, j, 0) = in(i, j, 0);
                for (int k = 1; k < TypeParam::k_size(); ++k)
                    refv(i, j, k) = in(i, j, k) + refv(i, j, k - 1);
            }
        TypeParam::verify(ref, field);
    }

    struct shift_acc_backward_fill_and_flush {
        using in = inout_accessor<0, extent<0, 0, 0, 0, 0, 1>>;
        using param_list = make_param_list<in>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::modify<0, -1>) {
            eval(in()) = eval(in()) + eval(in(0, 0, 1));
        }
        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
            eval(in()) = eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_fill_and_flush, backward) {
        auto field = TypeParam::make_storage(in);
        auto spec = [](auto in) {
            return execute_backward()
                .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in)
                .stage(shift_acc_backward_fill_and_flush(), in);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), field);
        auto ref = TypeParam::make_storage();
        auto refv = ref->host_view();
        for (int i = 0; i < TypeParam::d(0); ++i)
            for (int j = 0; j < TypeParam::d(1); ++j) {
                refv(i, j, TypeParam::k_size() - 1) = in(i, j, TypeParam::k_size() - 1);
                for (int k = TypeParam::k_size() - 2; k >= 0; --k)
                    refv(i, j, k) = refv(i, j, k + 1) + in(i, j, k);
            }
        TypeParam::verify(ref, field);
    }

    struct copy_fill {
        using in = inout_accessor<0>;
        using param_list = make_param_list<in>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull) {
            eval(in()) = eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_fill_and_flush, copy_forward) {
        auto field = TypeParam::make_storage(in);
        auto spec = [](auto in) {
            return execute_forward()
                .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in)
                .stage(copy_fill(), in);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), field);
        TypeParam::verify(in, field);
    }

    struct scale_fill {
        using in = inout_accessor<0>;
        using param_list = make_param_list<in>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull) {
            eval(in()) = 2 * eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_fill_and_flush, scale_forward) {
        auto field = TypeParam::make_storage(in);
        auto spec = [](auto in) {
            return execute_forward()
                .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in)
                .stage(scale_fill(), in);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), field);
        TypeParam::verify([](int i, int j, int k) { return 2 * in(i, j, k); }, field);
    }
} // namespace
