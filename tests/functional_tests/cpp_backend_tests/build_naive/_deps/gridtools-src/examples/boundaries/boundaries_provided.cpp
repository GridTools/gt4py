/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file This file contains several examples of using boundary conditions classes provided by GridTools itself.

    They are:

    - copy_boundary, that takes 2 or 3 fields, and copy the values at
      the boundary of the last one into the others;

    - zero_boundary, that set the boundary of the fields (maximum 3 in
      current implementation) to the default constructed value of the
      data_store value type;

    - value_boundary, that set the boundary to a specified value.

    We are using helper functions to show how to use them and a simple
    code to check correctness.
 */
#include <iostream>

#include <gridtools/boundaries/boundary.hpp>
#include <gridtools/boundaries/copy.hpp>
#include <gridtools/boundaries/value.hpp>
#include <gridtools/boundaries/zero.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/gcl/low_level/arch.hpp>
#include <gridtools/storage/builder.hpp>

#ifdef GT_CUDACC
#include <gridtools/storage/gpu.hpp>
using storage_traits_t = gridtools::storage::gpu;
using gcl_arch_t = gridtools::gcl::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_traits_t = gridtools::storage::cpu_ifirst;
using gcl_arch_t = gridtools::gcl::cpu;
#endif

namespace gt = gridtools;
namespace bd = gt::boundaries;

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " dimx dimy dimz\n"
                     " where args are integer sizes of the data fields"
                  << std::endl;
        return EXIT_FAILURE;
    }

    using uint_t = unsigned;

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    // Definition of the actual data fields that are used for input/output
    auto storage_builder = gt::storage::builder<storage_traits_t>.type<int>().dimensions(d1, d2, d3).halos(1, 1, 1);

    auto in_s = storage_builder.name("in").initializer([](int i, int j, int k) { return i + j + k; }).build();
    auto out_s = storage_builder.name("in").value(0).build();

    /* Defintion of the boundaries of the storage. We use halo_descriptor, that are used also in the current
       communication library. We plan to use a better structure in the future. The halo descriptor contains 5 numbers:
       - The halo in the minus direction
       - The halo in the plus direction
       - The begin of the inner region
       - The end (inclusive) of the inner region
       - The total length if the dimension.

       You need 3 halo descriptors, one per dimension.
    */
    gt::array<gt::halo_descriptor, 3> halos;
    halos[0] = gt::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gt::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gt::halo_descriptor(1, 1, 1, d3 - 2, d3);

    bool error = false;
    {
        bd::boundary<bd::copy_boundary, gcl_arch_t>(halos, bd::copy_boundary{}).apply(out_s, in_s);

        // making the views to access and check correctness
        auto in = in_s->const_host_view();
        auto out = out_s->const_host_view();

        for (int i = 0; i < d1; ++i) {
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3; ++k) {
                    // check outer surfaces of the cube
                    if ((i == 0 || i == d1 - 1) || (j == 0 || j == d2 - 1) || (k == 0 || k == d3 - 1)) {
                        error |= out(i, j, k) != i + j + k;
                        error |= out(i, j, k) != i + j + k;
                    } else {
                        error |= out(i, j, k) != 0;
                        error |= in(i, j, k) != i + j + k;
                    }
                }
            }
        }
    }

    {
        bd::boundary<bd::zero_boundary, gcl_arch_t>(halos, bd::zero_boundary{}).apply(out_s);

        // making the views to access and check correctness
        auto out = out_s->const_host_view();

        for (int i = 0; i < d1; ++i) {
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3; ++k) {
                    // check outer surfaces of the cube
                    if ((i == 0 || i == d1 - 1) || (j == 0 || j == d2 - 1) || (k == 0 || k == d3 - 1)) {
                        error |= out(i, j, k) != 0;
                    } else {
                        error |= out(i, j, k) != 0;
                    }
                }
            }
        }
    }

    {
        bd::boundary<bd::value_boundary<int>, gcl_arch_t>(halos, bd::value_boundary<int>{42}).apply(out_s);

        // making the views to access and check correctness
        auto out = out_s->const_host_view();

        for (int i = 0; i < d1; ++i) {
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3; ++k) {
                    // check outer surfaces of the cube
                    if ((i == 0 || i == d1 - 1) || (j == 0 || j == d2 - 1) || (k == 0 || k == d3 - 1)) {
                        error |= out(i, j, k) != 42;
                    } else {
                        error |= out(i, j, k) != 0;
                    }
                }
            }
        }
    }

    if (error) {
        std::cout << "TEST failed.\n";
    } else {
        std::cout << "TEST passed.\n";
    }

    return error;
}
