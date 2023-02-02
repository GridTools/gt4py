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

#include "div_functors.hpp"
#include "operators_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace ico_operators;

    GT_REGRESSION_TEST(div_reduction_into_scalar, icosahedral_test_environment<2>, stencil_backend_t) {
        auto spec = [](auto in_edges, auto edge_length, auto cell_area_reciprocal, auto out) {
            GT_DECLARE_ICO_TMP((array<typename TypeParam ::float_t, 3>), cells, weights);
            return execute_parallel()
                .ij_cached(weights)
                .stage(div_prep_functor(), edge_length, cell_area_reciprocal, weights)
                .stage(div_functor_reduction_into_scalar(), in_edges, weights, out);
        };
        operators_repository repo = {TypeParam::d(0), TypeParam::d(1)};
        auto out = TypeParam ::icosahedral_make_storage(cells());
        run(spec,
            stencil_backend_t(),
            TypeParam ::make_grid(),
            TypeParam ::icosahedral_make_storage(edges(), repo.u),
            TypeParam ::icosahedral_make_storage(edges(), repo.edge_length),
            TypeParam ::icosahedral_make_storage(cells(), repo.cell_area_reciprocal),
            out);
        TypeParam ::verify(repo.div_u, out);
    }

    GT_REGRESSION_TEST(div_flow_convention, icosahedral_test_environment<2>, stencil_backend_t) {
        operators_repository repo = {TypeParam::d(0), TypeParam::d(1)};
        auto out = TypeParam ::icosahedral_make_storage(cells());
        run_single_stage(div_functor_flow_convention_connectivity(),
            stencil_backend_t(),
            TypeParam::make_grid(),
            TypeParam ::icosahedral_make_storage(edges(), repo.u),
            TypeParam ::icosahedral_make_storage(edges(), repo.edge_length),
            TypeParam ::icosahedral_make_storage(cells(), repo.cell_area_reciprocal),
            out);
        TypeParam ::verify(repo.div_u, out);
    }
} // namespace
