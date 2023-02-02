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

    using axis_t = axis<1>;
    using full_t = axis_t::full_interval;

    struct set_constant {
        using out = inout_accessor<0>;

        using param_list = make_param_list<out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = 1;
        }
    };

    struct copy_with_vertical_offset {
        using in = in_accessor<0, extent<0, 0, 0, 0, 0, 1>>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, full_t::modify<0, -1>) {
            eval(out()) = eval(in(0, 0, 1));
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, full_t::last_level) {
            eval(out()) = eval(in());
        }
    };

#ifndef GT_STENCIL_GPU_HORIZONTAL
    GT_REGRESSION_TEST(parallel_multistage_fusion, test_environment<>, stencil_backend_t) {
        auto out = TypeParam::make_storage();
        auto comp = [&out, grid = TypeParam::make_grid()] {
            const auto spec = [](auto out) {
                GT_DECLARE_TMP(typename TypeParam::float_t, tmp);
                return multi_pass(execute_parallel().stage(set_constant(), tmp),
                    execute_parallel().stage(copy_with_vertical_offset(), tmp, out));
            };
            run(spec, stencil_backend_t(), grid, out);
        };
        comp();
        TypeParam::verify([](int, int, int) { return 1; }, out);
    }
#endif
} // namespace
