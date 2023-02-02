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

#include "../../common/defs.hpp"
#include "../../common/halo_descriptor.hpp"
#include "../core/grid.hpp"
#include "axis.hpp"

namespace gridtools {
    namespace stencil {
        template <class Axis>
        core::grid<typename Axis::axis_interval_t> make_grid(int_t di, int_t dj, Axis const &axis) {
            return {0, di, 0, dj, axis.interval_sizes()};
        }
        template <class Axis>
        core::grid<typename Axis::axis_interval_t> make_grid(
            halo_descriptor const &direction_i, halo_descriptor const &direction_j, Axis const &axis) {
            return {(int_t)direction_i.begin(),
                (int_t)direction_i.end() + 1 - (int_t)direction_i.begin(),
                (int_t)direction_j.begin(),
                (int_t)direction_j.end() + 1 - (int_t)direction_j.begin(),
                axis.interval_sizes()};
        }
        inline core::grid<axis<1>::axis_interval_t> make_grid(int_t di, int_t dj, int_t dk) {
            return {0, di, 0, dj, {dk}};
        }
        inline core::grid<axis<1>::axis_interval_t> make_grid(
            halo_descriptor const &direction_i, halo_descriptor const &direction_j, int_t dk) {
            return {(int_t)direction_i.begin(),
                (int_t)direction_i.end() + 1 - (int_t)direction_i.begin(),
                (int_t)direction_j.begin(),
                (int_t)direction_j.end() + 1 - (int_t)direction_j.begin(),
                {dk}};
        }
    } // namespace stencil
} // namespace gridtools
