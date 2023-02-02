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

#include <cassert>
#include <cmath>
#include <functional>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/frontend/icosahedral/location_type.hpp>

#include "neighbours_of.hpp"

class operators_repository {
    using cells = gridtools::stencil::icosahedral::cells;
    using edges = gridtools::stencil::icosahedral::edges;
    using vertices = gridtools::stencil::icosahedral::vertices;

    using fun_t = std::function<double(int, int, int, int)>;

    size_t m_d1, m_d2;

    const double PI = std::atan(1) * 4;

    template <class Location>
    double x(int i, int c) const {
        return (i + c * 1. / Location::value) / m_d1;
    }

    double y(int j) const { return j * 1. / m_d2; }

  public:
    fun_t u = [this](int i, int j, int k, int c) {
        auto t = PI * (x<edges>(i, c) + 1.5 * y(j));
        return k + 2 * (2 + cos(t) + sin(2 * t));
    };

    fun_t edge_length = [this](int i, int j, int, int c) {
        auto t = PI * (x<edges>(i, c) + 1.5 * y(j));
        return 2.95 + (2 + cos(t) + sin(2 * t)) / 4;
    };

    fun_t edge_length_reciprocal = [this](int i, int j, int k, int c) { return 1 / edge_length(i, j, k, c); };

    fun_t cell_area_reciprocal = [this](int i, int j, int, int c) {
        auto xx = x<cells>(i, c);
        auto yy = y(j);
        return 1 / (2.53 + (2 + cos(PI * (1.5 * xx + 2.5 * yy)) + sin(2 * PI * (xx + 1.5 * yy))) / 4);
    };

    fun_t dual_area_reciprocal = [this](int i, int j, int, int c) {
        auto xx = x<vertices>(i, c);
        auto yy = y(j);
        return 1 / (1.1 + (2 + cos(PI * (1.5 * xx + yy)) + sin(1.5 * PI * (xx + 1.5 * yy))) / 4);
    };

    fun_t dual_edge_length = [this](int i, int j, int, int c) {
        auto xx = x<edges>(i, c);
        auto yy = y(j);
        return 2.2 + (2 + cos(PI * (xx + 2.5 * yy)) + sin(2 * PI * (xx + 3.5 * yy))) / 4;
    };

    fun_t dual_edge_length_reciprocal = [this](int i, int j, int k, int c) { return 1 / dual_edge_length(i, j, k, c); };

    fun_t div_u = [this](int i, int j, int k, int c) {
        double res = 0;
        int e = 0;
        for (auto &&neighbour : gridtools::neighbours_of<cells, edges>(i, j, k, c)) {
            res += (c == 0 ? 1 : -1) * neighbour.call(u) * neighbour.call(edge_length);
            ++e;
        }
        return res * cell_area_reciprocal(i, j, k, c);
    };

    fun_t curl_u = [this](int i, int j, int k, int c) {
        double res = 0;
        int e = 0;
        for (auto &&neighbour : gridtools::neighbours_of<vertices, edges>(i, j, k, c)) {
            res += (e % 2 ? 1 : -1) * neighbour.call(u) * neighbour.call(dual_edge_length);
            ++e;
        }
        return res * dual_area_reciprocal(i, j, k, c);
    };

    fun_t lap = [this](int i, int j, int k, int c) {
        auto neighbours_ec = gridtools::neighbours_of<edges, cells>(i, j, k, c);
        assert(neighbours_ec.size() == 2);
        auto grad_n =
            (neighbours_ec[1].call(div_u) - neighbours_ec[0].call(div_u)) * dual_edge_length_reciprocal(i, j, k, c);

        auto neighbours_vc = gridtools::neighbours_of<edges, vertices>(i, j, k, c);
        assert(neighbours_vc.size() == 2);
        auto grad_tau =
            (neighbours_vc[1].call(curl_u) - neighbours_vc[0].call(curl_u)) * edge_length_reciprocal(i, j, k, c);
        return grad_n - grad_tau;
    };

    operators_repository(size_t d1, size_t d2) : m_d1(d1), m_d2(d2) {}
};
