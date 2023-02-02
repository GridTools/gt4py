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

    struct curl_prep_functor {
        using dual_area_reciprocal = in_accessor<0, vertices>;
        using dual_edge_length = in_accessor<1, edges, extent<-1, 0, -1, 0>>;
        using weights = inout_accessor<2, vertices>;

        using param_list = make_param_list<dual_area_reciprocal, dual_edge_length, weights>;
        using location = vertices;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            auto &&out = eval(weights());
            auto &&reciprocal = eval(dual_area_reciprocal());
            int e = 0;
            eval.for_neighbors(
                [&](auto len) {
                    out[e] = (e % 2 ? 1 : -1) * len * reciprocal;
                    ++e;
                },
                dual_edge_length());
        }
    };

    struct curl_functor_weights {
        using in_edges = in_accessor<0, edges, extent<-1, 0, -1, 0>>;
        using weights = in_accessor<1, vertices>;
        using out_vertices = inout_accessor<2, vertices>;

        using param_list = make_param_list<in_edges, weights, out_vertices>;
        using location = vertices;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            auto &&out = eval(out_vertices()) = 0;
            auto &&w = eval(weights());
            int e = 0;
            eval.for_neighbors([&](auto in) { out += in * w[e++]; }, in_edges());
        }
    };

    struct curl_functor_flow_convention {
        using in_edges = in_accessor<0, edges, extent<-1, 0, -1, 0>>;
        using dual_area_reciprocal = in_accessor<1, vertices>;
        using dual_edge_length = in_accessor<2, edges, extent<-1, 0, -1, 0>>;
        using out_vertices = inout_accessor<3, vertices>;

        using param_list = make_param_list<in_edges, dual_area_reciprocal, dual_edge_length, out_vertices>;
        using location = vertices;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            std::decay_t<decltype(eval(out_vertices()))> t = 0;
            int e = 0;
            eval.for_neighbors(
                [&](auto in, auto len) {
                    t += (e % 2 ? 1 : -1) * in * len;
                    ++e;
                },
                in_edges(),
                dual_edge_length());
            eval(out_vertices()) = t * eval(dual_area_reciprocal());
        }
    };
} // namespace ico_operators
