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
#include <gridtools/stencil/positional.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct functor {
        using out = inout_accessor<0>;
        using i_pos = in_accessor<1>;
        using j_pos = in_accessor<2>;
        using k_pos = in_accessor<3>;
        using param_list = make_param_list<out, i_pos, j_pos, k_pos>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(i_pos()) + eval(j_pos()) + eval(k_pos());
        }
    };

    GT_REGRESSION_TEST(positional_stencil, test_environment<>, stencil_backend_t) {
        auto out = TypeParam::make_storage();
        run_single_stage(functor(),
            stencil_backend_t(),
            TypeParam::make_grid(),
            out,
            positional<dim::i>(),
            positional<dim::j>(),
            positional<dim::k>());
        TypeParam::verify([](int i, int j, int k) { return i + j + k; }, out);
    }
} // namespace
