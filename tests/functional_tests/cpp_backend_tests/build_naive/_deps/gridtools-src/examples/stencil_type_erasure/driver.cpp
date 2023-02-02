/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "interpolate_stencil.hpp"

#include <iostream>

bool verify(double weight, inputs in, outputs out) {
    auto in1_v = in.in1->const_host_view();
    auto in2_v = in.in2->const_host_view();
    auto out_v = out.out->const_host_view();

    // check consistency
    assert(in1_v.lengths() == out_v.lengths());
    assert(in2_v.lengths() == out_v.lengths());

    auto &&len = out_v.lengths();
    for (int k = 0; k < len[2]; ++k)
        for (int i = 0; i < len[0]; ++i)
            for (int j = 0; j < len[1]; ++j)
                if (weight * in1_v(i, j, k) + (1.0 - weight) * in2_v(i, j, k) - out_v(i, j, k) > 1e-8) {
                    std::cerr << "error in " << i << ", " << j << ", " << k << ": "
                              << "expected = " << weight * in1_v(i, j, k) + (1.0 - weight) * in2_v(i, j, k)
                              << ", out = " << out_v(i, j, k) << ", diff = "
                              << weight * in1_v(i, j, k) + (1.0 - weight) * in2_v(i, j, k) - out_v(i, j, k)
                              << std::endl;
                    return false;
                }
    return true;
}
int main(int argc, char **argv) {
    unsigned int d1, d2, d3;
    const double weight = 0.4;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    // Add dimensions to the storage builder
    const auto storage_builder = gridtools::storage::builder<storage_traits_t>.dimensions(d1, d2, d3).type<double>();

    // Now we describe the iteration space. In this simple example the iteration space is just described by the full
    // grid (no particular care has to be taken to describe halo points).
    auto grid = gridtools::stencil::make_grid(d1, d2, d3);

    // Create some data stores
    inputs in = {storage_builder.initializer([](int i, int j, int k) { return i + j + k; }).build(),
        storage_builder.initializer([](int i, int j, int k) { return 4 * i + 2 * j + k; }).build()};
    outputs out = {storage_builder.build()};

    // Use the wrapped computation
    make_interpolate_stencil(grid, weight)(in, out);

    if (!verify(weight, in, out)) {
        std::cerr << "Failure" << std::endl;
        return 1;
    }
    std::cout << "Success" << std::endl;
};
