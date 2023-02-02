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

#include <functional>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/frontend/make_grid.hpp>
#include <gridtools/storage/builder.hpp>

#ifdef USE_GPU
#include <gridtools/storage/gpu.hpp>
using storage_traits_t = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_traits_t = gridtools::storage::cpu_ifirst;
#endif

using data_store_t = decltype(gridtools::storage::builder<storage_traits_t>.dimensions(0, 0, 0).type<double>().build());

using grid_t = decltype(gridtools::stencil::make_grid(0, 0, 0));

struct inputs {
    data_store_t in1;
    data_store_t in2;
};
struct outputs {
    data_store_t out;
};

std::function<void(inputs, outputs)> make_interpolate_stencil(grid_t grid, double weight);
