/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend.
*/

#include <cstdlib>
#include <iostream>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GT_CUDACC
#include <gridtools/stencil/gpu.hpp>
#include <gridtools/storage/gpu.hpp>
using stencil_backend_t = gridtools::stencil::gpu<>;
using storage_traits_t = gridtools::storage::gpu;
#else
#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
using stencil_backend_t = gridtools::stencil::cpu_ifirst<>;
using storage_traits_t = gridtools::storage::cpu_ifirst;
#endif

namespace gt = gridtools;

// This is the stencil operator which copies the value from `in` to `out`.
struct copy_functor {
    using in = gt::stencil::cartesian::in_accessor<0>;
    using out = gt::stencil::cartesian::inout_accessor<1>;
    using param_list = gt::stencil::make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in());
    }
};

int main(int argc, char **argv) {
    int d1, d2, d3;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    // Add dimensions to the storage builder
    auto storage_builder = gt::storage::builder<storage_traits_t>.dimensions(d1, d2, d3);

    auto f = [](int i, int j, int k) { return i + j + k; };

    // Build input storage
    auto in = storage_builder.initializer(f).type<const double>().build();

    // Build the storage to collect the result
    auto out = storage_builder.type<double>().build();

    // Now we describe the iteration space. In this simple example the iteration space is just described by the full
    // grid (no particular care has to be taken to describe halo points).
    auto grid = gt::stencil::make_grid(d1, d2, d3);

    // Execute the computation
    gt::stencil::run_single_stage(copy_functor(), stencil_backend_t(), grid, in, out);

    // Compare the result
    auto view = out->const_host_view();
    for (int k = 0; k < d3; ++k)
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j)
                if (view(i, j, k) != f(i, j, k)) {
                    std::cerr << "error in " << i << ", " << j << ", " << k << ": "
                              << "actual = " << view(i, j, k) << ", expected = " << f(i, j, k) << std::endl;
                    return 1;
                }

    std::cout << "Success" << std::endl;
};
