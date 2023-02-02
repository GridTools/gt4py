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

    GT_ENVIRONMENT_TEST_SUITE(test_kcache_fill,
        (vertical_test_environment<0, axis_t>),
        stencil_backend_t,
        (double, inlined_params<6, 6, 2, 6, 2>));

    struct shift_acc_forward_fill {
        using in = in_accessor<0, extent<0, 0, 0, 0, -1, 1>>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
            eval(out()) = eval(in()) + eval(in(0, 0, 1));
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, -1>) {
            eval(out()) = eval(in(0, 0, -1)) + eval(in()) + eval(in(0, 0, 1));
        }
        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
            eval(out()) = eval(in(0, 0, -1)) + eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_fill, forward) {
        auto out = TypeParam::make_storage();
        auto spec = [](auto in, auto out) {
            return execute_forward().k_cached(cache_io_policy::fill(), in).stage(shift_acc_forward_fill(), in, out);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), TypeParam::make_storage(in), out);
        auto ref = TypeParam::make_storage();
        auto refv = ref->host_view();
        for (int i = 0; i < TypeParam::d(0); ++i) {
            for (int j = 0; j < TypeParam::d(1); ++j) {
                refv(i, j, 0) = in(i, j, 0) + in(i, j, 1);
                for (int k = 1; k < TypeParam::k_size() - 1; ++k)
                    refv(i, j, k) = in(i, j, k - 1) + in(i, j, k) + in(i, j, k + 1);
                refv(i, j, TypeParam::k_size() - 1) =
                    in(i, j, TypeParam::k_size() - 1) + in(i, j, TypeParam::k_size() - 2);
            }
        }
        TypeParam::verify(ref, out);
    }

    struct shift_acc_backward_fill {
        using in = in_accessor<0, extent<0, 0, 0, 0, -1, 1>>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
            eval(out()) = eval(in()) + eval(in(0, 0, -1));
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, -1>) {
            eval(out()) = eval(in(0, 0, 1)) + eval(in()) + eval(in(0, 0, -1));
        }
        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
            eval(out()) = eval(in()) + eval(in(0, 0, 1));
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_fill, backward) {
        auto out = TypeParam::make_storage();
        auto spec = [](auto in, auto out) {
            return execute_backward().k_cached(cache_io_policy::fill(), in).stage(shift_acc_backward_fill(), in, out);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), TypeParam::make_storage(in), out);
        auto ref = TypeParam::make_storage();
        auto refv = ref->host_view();
        for (int i = 0; i < TypeParam::d(0); ++i) {
            for (int j = 0; j < TypeParam::d(1); ++j) {
                refv(i, j, TypeParam::k_size() - 1) =
                    in(i, j, TypeParam::k_size() - 1) + in(i, j, TypeParam::k_size() - 2);
                for (int_t k = TypeParam::k_size() - 2; k >= 1; --k)
                    refv(i, j, k) = in(i, j, k + 1) + in(i, j, k) + in(i, j, k - 1);
                refv(i, j, 0) = in(i, j, 1) + in(i, j, 0);
            }
        }
        TypeParam::verify(ref, out);
    }

    struct copy_fill {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, kfull) {
            eval(out()) = eval(in());
        }
    };

    GT_ENVIRONMENT_TYPED_TEST(test_kcache_fill, fill_copy_forward) {
        auto out = TypeParam::make_storage();
        auto spec = [](auto in, auto out) {
            return execute_forward().k_cached(cache_io_policy::fill(), in).stage(copy_fill(), in, out);
        };
        run(spec, stencil_backend_t(), TypeParam::make_grid(), TypeParam::make_storage(in), out);
        TypeParam::verify(in, out);
    }
} // namespace
