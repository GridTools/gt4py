/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file
    This file contains several examples of using boundary conditions.
    The basic concept, here and in distributed boundaries and communication is the concept of "direction".

    In a 3D regular grid, which is where this implementation of the boundary condition library applies, we associate a
   3D axis system, and the cell indices (i,j,k) naturally lie on it. With this axis system the concept of "vector" can
   be defined to indicate distances and directions. Direction is the one thing we need here. Instead of using unitary
   vectors to indicate directions, as it is usually the case for Euclidean spaces, we use vectors whose components are
   -1, 0, and 1.  For example, (1, 1, 1) is the direction indicated by the unit vector (1,1,1)/sqrt(3).

    If we take the center of a 3D grid, then we can define 26 different directions {(i,j,k): i,j,k \in {-1, 0, 1}} \ {
   (0,0,0) } that identify the different faces, edges and corners of the cube to which the grid is topologically
   analogous with.

    THE MAIN IDEA:
    A boundary condition class specializes `operator()` to accept a  direction and when that direction is accessed, the
   data fields in the boundary corresponding to that direction can be accessed.
 */

#include <iostream>

#include <gridtools/boundaries/boundary.hpp>
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

using uint_t = unsigned;

/**
   This class specifies how to apply boundary conditions.

   For all directions, apart from (0,-1,0), (-1,-1,0), and (-1,-1,-1) it writes the values associated with the object in
   the boundary of the first field, otherwise it copies the values of the second field from a shifted position.

   The directions here are specified at compile time, and instead of  using numbers we use minus_, plus_ and zero_, in
   order to be more explicit.

   The second field is not modified.
 */
template <typename T>
struct direction_bc_input {
    T value;

    GT_FUNCTION
    direction_bc_input() : value(1) {}

    GT_FUNCTION
    direction_bc_input(T v) : value(v) {}

    // This is the primary template and it will be picked when all the other specializations below fail to be selected.
    // It is important for debugging to note that, if a needed specialization i missing, this version will be selected.
    template <typename Direction, typename DataField0, typename DataField1>
    GT_FUNCTION void operator()(
        Direction, DataField0 &data_field0, DataField1 const &, uint_t i, uint_t j, uint_t k) const {
        data_field0(i, j, k) = value;
    }

    template <bd::sign I, bd::sign K, typename DataField0, typename DataField1>
    GT_FUNCTION void operator()(bd::direction<I, bd::minus_, K>,
        DataField0 &data_field0,
        DataField1 const &data_field1,
        uint_t i,
        uint_t j,
        uint_t k) const {
        data_field0(i, j, k) = data_field1(i, j + 1, k);
    }

    template <bd::sign K, typename DataField0, typename DataField1>
    GT_FUNCTION void operator()(bd::direction<bd::minus_, bd::minus_, K>,
        DataField0 &data_field0,
        DataField1 const &data_field1,
        uint_t i,
        uint_t j,
        uint_t k) const {
        data_field0(i, j, k) = data_field1(i + 1, j + 1, k);
    }

    template <typename DataField0, typename DataField1>
    GT_FUNCTION void operator()(bd::direction<bd::minus_, bd::minus_, bd::minus_>,
        DataField0 &data_field0,
        DataField1 const &data_field1,
        uint_t i,
        uint_t j,
        uint_t k) const {
        data_field0(i, j, k) = data_field1(i + 1, j + 1, k + 1);
    }
};

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " dimx dimy dimz\n"
                     " where args are integer sizes of the data fields"
                  << std::endl;
        return EXIT_FAILURE;
    }

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    // Definition of the actual data fields that are used for input/output
    auto storage_builder = gt::storage::builder<storage_traits_t>.type<int>().dimensions(d1, d2, d3).halos(1, 1, 1);

    auto in_s = storage_builder.name("in").initializer([](int i, int j, int k) { return i + j + k; }).build();
    auto out_s = storage_builder.name("in").value(0).build();

    /* Definition of the boundaries of the storage. We use halo_descriptor, that are used also in the current
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

    // Here we apply the boundary conditions to the fields created earlier with the class above. GridTools provides
    // default boundary classes to copy fields and to set constant values to the boundaries of fields.
    bd::boundary<direction_bc_input<uint_t>, gcl_arch_t>(halos, direction_bc_input<uint_t>(42)).apply(out_s, in_s);

    // making the views to access and check correctness
    auto in = in_s->const_host_view();
    auto out = out_s->const_host_view();

    bool error = false;

    // check edge column
    if (out(0, 0, 0) != in(1, 1, 1)) {
        std::cout << "Error: out(0, 0, 0) == " << out(0, 0, 0) << " != in(1,1,1) = " << in(1, 1, 1) << "\n";
        error = true;
    }
    for (uint_t k = 1; k < d3; ++k) {
        if (out(0, 0, k) != in(1, 1, k)) {
            std::cout << "Error: out(0, 0, " << k << ") == " << out(0, 0, 0) << " != in(1, 1, " << k
                      << ") = " << in(1, 1, k) << "\n";
            error = true;
        }
    }

    // check j==0 i>0 surface
    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t k = 0; k < d3; ++k) {
            if (out(i, 0, k) != in(i, 1, k)) {
                std::cout << "Error: out(0, 0, 0) == " << out(i, 0, k) << " != in(" << i + 1 << ",0," << k + 1
                          << ") == " << in(i + 1, 0, k + 1) << "\n";
                error = true;
            }
        }
    }

    // check outer domain
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                // check outer surfaces of the cube
                if (((i == 0 || i == d1 - 1) && j > 0) || (j > 0 && (k == 0 || k == d3 - 1))) {
                    if (out(i, j, k) != 42) {
                        std::cout << "Error: out(0, 0, 0) == " << out(i, 0, k) << " != 42"
                                  << "\n";

                        error = true;
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
