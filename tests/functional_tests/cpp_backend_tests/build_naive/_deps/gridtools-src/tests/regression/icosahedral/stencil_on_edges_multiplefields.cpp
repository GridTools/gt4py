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

    struct test_on_edges_functor {
        using in1 = in_accessor<0, edges, extent<1, -1, 1, -1>>;
        using in2 = in_accessor<1, edges, extent<1, -1, 1, -1>>;
        using out = inout_accessor<2, edges>;
        using param_list = make_param_list<in1, in2, out>;
        using location = edges;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            using float_t = std::decay_t<decltype(eval(out()))>;
            float_t res = 0;
            eval.for_neighbors([&](auto in1, auto in2) { res += in1 + in2 * float_t(.1); }, in1(), in2());
            eval(out()) = res;
        }
    };

    GT_REGRESSION_TEST(stencil_on_edges_multiplefields, icosahedral_test_environment<1>, stencil_backend_t) {
        auto in1 = [](int_t i, int_t j, int_t k, int_t c) { return i + j + k + c; };
        auto in2 = [](int_t i, int_t j, int_t k, int_t c) { return i / 2 + j / 2 + k / 2 + c; };
        auto ref = [=](int_t i, int_t j, int_t k, int_t c) {
            typename TypeParam::float_t res = 0;
            for (auto &&item : neighbours_of<edges, edges>(i, j, k, c))
                res += item.call(in1) + .1 * item.call(in2);
            return res;
        };
        auto out = TypeParam::icosahedral_make_storage(edges());
        auto comp = [grid = TypeParam::make_grid(),
                        in1 = TypeParam::icosahedral_make_storage(edges(), in1),
                        in2 = TypeParam::icosahedral_make_storage(edges(), in2),
                        &out] { run_single_stage(test_on_edges_functor(), stencil_backend_t(), grid, in1, in2, out); };
        comp();
        TypeParam::verify(ref, out);
        TypeParam::benchmark("stencil_on_edges_multiplefields", comp);
    }
} // namespace
