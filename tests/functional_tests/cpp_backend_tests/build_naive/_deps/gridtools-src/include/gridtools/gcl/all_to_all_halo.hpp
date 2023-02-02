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

#include "../common/halo_descriptor.hpp"
#include "low_level/Generic_All_to_All.hpp"
#include "low_level/data_types_mapping.hpp"

namespace gridtools {
    namespace gcl {
        template <typename vtype, typename pgrid>
        struct all_to_all_halo {
            /** Type of the elements to be exchanged.
             */
            typedef vtype value_type;

            /** Type of the processing grid used for data exchange
             */
            typedef pgrid grid_type;

            /** Number of dimensions of the computing grid
             */
            static const int ndims = grid_type::ndims;

          private:
            const grid_type proc_grid;
            all_to_all<value_type> a2a;

          public:
            /** Constructor that takes the computing grid and initializes the
                patern.
             */
            all_to_all_halo(grid_type const &g) : proc_grid(g), a2a(proc_grid.size()) {}

            /** Constructor that takes the computing grid and initializes the
                pattern. It also takes a communicator that is inside the processor
                grid, if different from MPI_COMM_WORLD
             */
            all_to_all_halo(grid_type const &g, MPI_Comm c) : proc_grid(g), a2a(proc_grid.size(), c) {}

            /** This function takes an array or vector of halos (sorted by
                decreasing strides) (size equal to ndims), the pointer to the
                data and the coordinated of the receiving processors and
                prepare the pattern to send that sub-array to that processor.

                \tparam arraytype1 type of the array of halos. This is required to have only the operator[] and the
               method size() \tparam arraytype2 type of the array of coordinates of the destination process. This is
               required to have only the operator[]

                \param field Pointer to the data do be sent
                \param halo_block or vector of type arraytype1 that contains the description of the data to be sent
                \param coords Array of vector of absolute coordinates of the process that will receive the data
             */
            template <typename arraytype1, typename arraytype2>
            void register_block_to(value_type *field, arraytype1 const &halo_block, arraytype2 const &coords) {
                a2a.to[proc_grid.abs_proc(coords)] =
                    packet<value_type>(make_datatype<value_type>::make(halo_block), field);
            }

            /** This function takes an array or vector of halos (sorted by
                decreasing strides) (size equal to ndims), the pointer to the
                data and the coordinated of the receiving processors and
                prepare the pattern to receive that sub-array from that
                processor.

                \tparam arraytype1 type of the array of halos. This is required to have only the operator[] and the
               method size() \tparam arraytype2 type of the array of coordinates of the destination process. This is
               required to have only the operator[]

                \param field Pointer to the tata do be received
                \param halo_block or vector of type arraytype1 that contains the description of the data to be received
                \param coords Array of vector of absolute coordinates of the process that from where the data will be
               received
             */
            template <typename arraytype1, typename arraytype2>
            void register_block_from(value_type *field, arraytype1 const &halo_block, arraytype2 const &coords) {
                a2a.from[proc_grid.abs_proc(coords)] =
                    packet<value_type>(make_datatype<value_type>::make(halo_block), field);
            }

            /** This method prepare the pattern to be ready to execute
             */
            void setup() { a2a.setup(); }

            /**
             * Method to post receives
             */
            void post_receives() { a2a.post_receives(); }

            /**
             * Method to send data
             */
            void do_sends() { a2a.do_sends(); }

            /** This method starts the data exchange
             */
            void start_exchange() { a2a.start_exchange(); }

            /** This method waits for the data to arrive and be unpacked
             */
            void wait() { a2a.wait(); }
        };
    } // namespace gcl
} // namespace gridtools
