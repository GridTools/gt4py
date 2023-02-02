/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include <gridtools/stencil/icosahedral.hpp>

namespace ico_operators {

    using namespace gridtools;
    using namespace stencil;
    using namespace icosahedral;

    struct div_prep_functor {
        using edge_length = in_accessor<0, edges, extent<0, 1, 0, 1>>;
        using cell_area_reciprocal = in_accessor<1, cells>;
        using weights = inout_accessor<2, cells>;

        using param_list = make_param_list<edge_length, cell_area_reciprocal, weights>;
        using location = cells;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            auto &&out = eval(weights());
            auto coeff = (Eval::color == 0 ? 1 : -1) * eval(cell_area_reciprocal());
            int e = 0;
            eval.for_neighbors([&](auto len) { out[e++] = coeff * len; }, edge_length());
        }
    };

    struct div_functor_reduction_into_scalar {
        using in_edges = in_accessor<0, edges, extent<0, 1, 0, 1>>;
        using weights = in_accessor<1, cells>;
        using out_cells = inout_accessor<2, cells>;

        using param_list = make_param_list<in_edges, weights, out_cells>;
        using location = cells;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            auto &&out = eval(out_cells()) = 0;
            auto &&w = eval(weights());
            int e = 0;
            eval.for_neighbors([&](auto in) { out += in * w[e++]; }, in_edges());
        }
    };

    struct div_functor_flow_convention_connectivity {
        using in_edges = in_accessor<0, edges, extent<0, 1, 0, 1>>;
        using edge_length = in_accessor<1, edges, extent<0, 1, 0, 1>>;
        using cell_area_reciprocal = in_accessor<2, cells>;
        using out_cells = inout_accessor<3, cells>;

        using param_list = make_param_list<in_edges, edge_length, cell_area_reciprocal, out_cells>;
        using location = cells;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            std::decay_t<decltype(eval(out_cells()))> t = 0;
            eval.for_neighbors([&](auto in, auto len) { t += in * len; }, in_edges(), edge_length());
            eval(out_cells()) = (Eval::color == 0 ? 1 : -1) * t * eval(cell_area_reciprocal());
        }
    };
} // namespace ico_operators
