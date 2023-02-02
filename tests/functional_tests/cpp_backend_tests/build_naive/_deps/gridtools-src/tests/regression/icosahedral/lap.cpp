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

#include "curl_functors.hpp"
#include "div_functors.hpp"
#include "operators_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace ico_operators;

    struct lap_functor {
        typedef in_accessor<0, cells, extent<-1, 0, -1, 0>> in_cells;
        typedef in_accessor<1, edges> dual_edge_length_reciprocal;
        typedef in_accessor<2, vertices, extent<0, 1, 0, 1>> in_vertices;
        typedef in_accessor<3, edges> edge_length_reciprocal;
        typedef inout_accessor<4, edges> out_edges;
        using param_list =
            make_param_list<in_cells, dual_edge_length_reciprocal, in_vertices, edge_length_reciprocal, out_edges>;
        using location = edges;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            std::decay_t<decltype(eval(dual_edge_length_reciprocal()))> grad_n = 0;
            int e = 0;
            eval.for_neighbors([&](auto in) { grad_n += (e++ ? 1 : -1) * in; }, in_cells());
            grad_n *= eval(dual_edge_length_reciprocal());

            std::decay_t<decltype(eval(out_edges()))> grad_tau = 0;
            e = 0;
            eval.for_neighbors([&](auto in) { grad_tau += (e++ ? 1 : -1) * in; }, in_vertices());
            grad_tau *= eval(edge_length_reciprocal());

            eval(out_edges()) = grad_n - grad_tau;
        }
    };

    GT_REGRESSION_TEST(lap_weights, icosahedral_test_environment<2>, stencil_backend_t) {
        using float_t = typename TypeParam ::float_t;
        auto spec = [](auto edge_length,
                        auto cell_area_reciprocal,
                        auto dual_area_reciprocal,
                        auto dual_edge_length,
                        auto in_edges,
                        auto dual_edge_length_reciprocal,
                        auto edge_length_reciprocal,
                        auto out) {
            GT_DECLARE_ICO_TMP(float_t, cells, div_on_cells);
            GT_DECLARE_ICO_TMP(float_t, vertices, curl_on_vertices);
            GT_DECLARE_ICO_TMP((array<float_t, 3>), cells, div_weights);
            GT_DECLARE_ICO_TMP((array<float_t, 6>), vertices, curl_weights);
            // sorry, curl_weights doesn't fit the ij_cache on daint gpu :(
            return execute_parallel()
                .ij_cached(div_on_cells, curl_on_vertices, div_weights /*, curl_weights*/)
                .stage(div_prep_functor(), edge_length, cell_area_reciprocal, div_weights)
                .stage(curl_prep_functor(), dual_area_reciprocal, dual_edge_length, curl_weights)
                .stage(div_functor_reduction_into_scalar(), in_edges, div_weights, div_on_cells)
                .stage(curl_functor_weights(), in_edges, curl_weights, curl_on_vertices)
                .stage(lap_functor(),
                    div_on_cells,
                    dual_edge_length_reciprocal,
                    curl_on_vertices,
                    edge_length_reciprocal,
                    out);
        };
        operators_repository repo = {TypeParam::d(0), TypeParam::d(1)};
        auto out = TypeParam::icosahedral_make_storage(edges());
        run(spec,
            stencil_backend_t(),
            TypeParam::make_grid(),
            TypeParam::icosahedral_make_storage(edges(), repo.edge_length),
            TypeParam::icosahedral_make_storage(cells(), repo.cell_area_reciprocal),
            TypeParam::icosahedral_make_storage(vertices(), repo.dual_area_reciprocal),
            TypeParam::icosahedral_make_storage(edges(), repo.dual_edge_length),
            TypeParam::icosahedral_make_storage(edges(), repo.u),
            TypeParam::icosahedral_make_storage(edges(), repo.dual_edge_length_reciprocal),
            TypeParam::icosahedral_make_storage(edges(), repo.edge_length_reciprocal),
            out);
        TypeParam::verify(TypeParam::icosahedral_make_storage(edges(), repo.lap), out);
    }

    GT_REGRESSION_TEST(lap_flow_convention, icosahedral_test_environment<2>, stencil_backend_t) {
        using float_t = typename TypeParam ::float_t;
        auto spec = [](auto in_edges,
                        auto edge_length,
                        auto cell_area_reciprocal,
                        auto dual_area_reciprocal,
                        auto dual_edge_length,
                        auto dual_edge_length_reciprocal,
                        auto edge_length_reciprocal,
                        auto out) {
            GT_DECLARE_ICO_TMP(float_t, cells, div_on_cells);
            GT_DECLARE_ICO_TMP(float_t, vertices, curl_on_vertices);
            return execute_parallel()
                .ij_cached(div_on_cells, curl_on_vertices)
                .stage(div_functor_flow_convention_connectivity(),
                    in_edges,
                    edge_length,
                    cell_area_reciprocal,
                    div_on_cells)
                .stage(
                    curl_functor_flow_convention(), in_edges, dual_area_reciprocal, dual_edge_length, curl_on_vertices)
                .stage(lap_functor(),
                    div_on_cells,
                    dual_edge_length_reciprocal,
                    curl_on_vertices,
                    edge_length_reciprocal,
                    out);
        };
        operators_repository repo = {TypeParam::d(0), TypeParam::d(1)};
        auto out = TypeParam::icosahedral_make_storage(edges());
        run(spec,
            stencil_backend_t(),
            TypeParam::make_grid(),
            TypeParam::icosahedral_make_storage(edges(), repo.u),
            TypeParam::icosahedral_make_storage(edges(), repo.edge_length),
            TypeParam::icosahedral_make_storage(cells(), repo.cell_area_reciprocal),
            TypeParam::icosahedral_make_storage(vertices(), repo.dual_area_reciprocal),
            TypeParam::icosahedral_make_storage(edges(), repo.dual_edge_length),
            TypeParam::icosahedral_make_storage(edges(), repo.dual_edge_length_reciprocal),
            TypeParam::icosahedral_make_storage(edges(), repo.edge_length_reciprocal),
            out);
        TypeParam::verify(TypeParam::icosahedral_make_storage(edges(), repo.lap), out);
    }
} // namespace
