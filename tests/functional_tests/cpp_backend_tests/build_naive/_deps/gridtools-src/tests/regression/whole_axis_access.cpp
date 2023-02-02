/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <type_traits>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/positional.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;
    using namespace literals;

    struct functor {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<>, 4>;
        using k_pos = in_accessor<2>;
        using param_list = make_param_list<out, in, k_pos>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            auto k = eval(k_pos());
            std::decay_t<decltype(eval(out()))> res = 0;
            for (int kk = 0; kk < k; ++kk)
                res += eval(in(0, 0, 0, kk));
            eval(out()) = res;
        }
    };

    GT_REGRESSION_TEST(whole_axis_access, test_environment<>, stencil_backend_t) {
        auto in = [](int i, int j, int k) { return i + j + k; };
        auto out = TypeParam::make_storage();
        run_single_stage(functor(),
            stencil_backend_t(),
            TypeParam::make_grid(),
            out,
            sid::rename_dimensions<dim::k, decltype(3_c)>(TypeParam::make_storage(in)),
            positional<dim::k>());
        TypeParam::verify(
            [in](int i, int j, int k) {
                int res = 0;
                for (int kk = 0; kk < k; ++kk)
                    res += in(i, j, kk);
                return res;
            },
            out);
    }
} // namespace
