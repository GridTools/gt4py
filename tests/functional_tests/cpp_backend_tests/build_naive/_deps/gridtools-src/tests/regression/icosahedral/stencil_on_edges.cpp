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
        using in = in_accessor<0, edges, extent<-1, 1, -1, 1>>;
        using out = inout_accessor<1, edges>;
        using param_list = make_param_list<in, out>;
        using location = edges;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            std::decay_t<decltype(eval(out()))> res = 0;
            eval.for_neighbors([&](auto in) { res += in; }, in());
            eval(out()) = res;
        }
    };

    GT_REGRESSION_TEST(stencil_on_edges, icosahedral_test_environment<1>, stencil_backend_t) {
        auto in = [](int_t i, int_t j, int_t k, int_t c) { return i + j + k + c; };
        auto ref = [&](int_t i, int_t j, int_t k, int_t c) {
            typename TypeParam::float_t res = 0;
            for (auto &&item : neighbours_of<edges, edges>(i, j, k, c))
                res += item.call(in);
            return res;
        };
        auto out = TypeParam ::icosahedral_make_storage(edges());
        run_single_stage(test_on_edges_functor(),
            stencil_backend_t(),
            TypeParam ::make_grid(),
            TypeParam ::icosahedral_make_storage(edges(), in),
            out);
        TypeParam ::verify(ref, out);
    }
} // namespace
