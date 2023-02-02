/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct lap {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 1, -1, 1>>;
        using param_list = make_param_list<out, in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            eval(out()) = 4 * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
        }
    };

    GT_REGRESSION_TEST(laplacian, test_environment<1>, stencil_backend_t) {
        auto in = [](int_t, int_t, int_t) { return -1.; };
        auto ref = [in](int_t i, int_t j, int_t k) {
            return 4 * in(i, j, k) - (in(i + 1, j, k) + in(i, j + 1, k) + in(i - 1, j, k) + in(i, j - 1, k));
        };
        auto out = TypeParam::make_storage();
        run_single_stage(lap(), stencil_backend_t(), TypeParam::make_grid(), out, TypeParam::make_storage(in));
        TypeParam::verify(ref, out);
    }
} // namespace
