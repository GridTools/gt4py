/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/icosahedral.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

#include "neighbours_of.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace icosahedral;

    template <int Color>
    using sign = integral_constant<int, Color == 0 ? -1 : 1>;

    struct on_cells_color_functor {
        using in = in_accessor<0, cells, extent<1, -1, 1, -1>>;
        using out = inout_accessor<1, cells>;
        using param_list = make_param_list<in, out>;
        using location = cells;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            std::decay_t<decltype(eval(out()))> res = 0;
            eval.for_neighbors([&](auto in) { res += sign<Eval::color>::value * in; }, in());
            eval(out()) = res;
        }
    };

    GT_REGRESSION_TEST(stencil_on_cells_with_color, icosahedral_test_environment<1>, stencil_backend_t) {
        auto in = [](int_t i, int_t j, int_t k, int_t c) { return i + j + k + c; };
        auto ref = [&](int_t i, int_t j, int_t k, int_t c) {
            typename TypeParam::float_t res = {};
            for (auto &&item : neighbours_of<cells, cells>(i, j, k, c))
                if (c == 0)
                    res -= item.call(in);
                else
                    res += item.call(in);
            return res;
        };
        auto out = TypeParam ::icosahedral_make_storage(cells());
        run_single_stage(on_cells_color_functor(),
            stencil_backend_t(),
            TypeParam ::make_grid(),
            TypeParam ::icosahedral_make_storage(cells(), in),
            out);
        TypeParam ::verify(ref, out);
    }
} // namespace
