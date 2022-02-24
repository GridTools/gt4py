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

#include <array>

#include <gridtools/common/integral_constant.hpp>

namespace simple_mesh {
    /*
     *          (2)
     *       1   2    3
     *   (1)  0     4   (3)
     *   11     (0)      5
     *   (6) 10      6  (4)
     *      9    8   7
     *          (5)
     */

    using namespace gridtools::literals;

    constexpr auto n_vertices = 7_c;
    constexpr auto n_edges = 12_c;

    constexpr std::array<int, 2> e2v[n_edges] = {
        {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 0}, {3, 5}, {4, 0}, {4, 5}, {5, 0}, {5, 6}, {6, 0}, {6, 1}};

    constexpr std::array<int, 6> v2e[n_vertices] = {{0, 2, 4, 6, 8, 10},
        {0, 1, 11, -1, -1, -1},
        {1, 2, 3, -1, -1, -1},
        {3, 4, 5, -1, -1, -1},
        {5, 6, 7, -1, -1, -1},
        {7, 8, 9, -1, -1, -1},
        {9, 10, 11, -1, -1, -1}};

} // namespace fgridtools::fn::simple_mesh
