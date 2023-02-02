/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/array.hpp>
#include <gridtools/stencil/icosahedral.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

#include "neighbours_of.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace icosahedral;

    struct test_on_edges_functor {
        using cell_area = in_accessor<0, cells, extent<-1, 1, -1, 1>>;
        using weight_edges = inout_accessor<1, cells>;
        using param_list = make_param_list<cell_area, weight_edges>;
        using location = cells;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            auto &&out = eval(weight_edges());
            auto focus = eval(cell_area());
            int i = 0;
            eval.for_neighbors([&](auto neighbor) { out[i++] = neighbor / focus; }, cell_area());
        }
    };

    GT_REGRESSION_TEST(stencil_manual_fold, icosahedral_test_environment<1>, stencil_backend_t) {
        using float_t = typename TypeParam::float_t;
        using weight_edges_t = array<float_t, 3>;

        auto in = [](int_t i, int_t j, int_t k, int_t c) -> float_t { return 1. + i + j + k + c; };
        auto ref = [&](int_t i, int_t j, int_t k, int_t c) -> weight_edges_t {
            auto val = [&](int e) -> float_t {
                return neighbours_of<cells, cells>(i, j, k, c)[e].call(in) / in(i, j, k, c);
            };
            return {val(0), val(1), val(2)};
        };
        auto out = TypeParam::template icosahedral_make_storage<weight_edges_t>(cells());
        auto comp = [&, grid = TypeParam::make_grid(), in = TypeParam::icosahedral_make_storage(cells(), in)] {
            run_single_stage(test_on_edges_functor(), stencil_backend_t(), grid, in, out);
        };
        comp();
        TypeParam::verify(ref, out, [](auto lhs, auto rhs) {
            for (size_t i = 0; i != rhs.size(); ++i)
                if (!expect_with_threshold(lhs[i], rhs[i]))
                    return false;
            return true;
        });
        TypeParam::benchmark("stencil_manual_fold", comp);
    }
} // namespace
