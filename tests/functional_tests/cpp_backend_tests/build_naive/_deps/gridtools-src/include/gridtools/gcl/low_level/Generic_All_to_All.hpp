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

#include <vector>

#include <mpi.h>

#include "../GCL.hpp"

/** \file

    This file contains some low level MPI wrappers that should not
    used by users directly, in principle. They have been thought to be
    a bridge to Level 2 interfaces of generic all to all with
    halos. Additional work is needed to provide a MPI-independent
    version of the all to all for level 3.
 */

namespace gridtools {
    namespace gcl {
        template <typename vtype>
        struct all_to_all;

        /** This struct describe an MPI datatype along with the pointer to
            the data. The datatype and the pointer must be known and build
            outside.

            \tparam T Value type of the data the pointer points to
         */
        template <typename T>
        struct packet {
            /** Type of the elements to be exchanged.
             */
            typedef T value_type;

          private:
            MPI_Datatype mpidt;
            value_type *ptr;
            MPI_Request send_r;
            MPI_Request recv_r;

          public:
            /** Default constructor set pointer to null. This is useful since
                the pointer value is used to determine if a message should be
                sent or not. If not null, a mpi send/receive is issued.
             */
            packet() : mpidt(), ptr(nullptr), send_r(), recv_r() {}

            /** This is the basic constructor. Given the MPI datatype and the
                pointer to data, the packet is constructed. If pointer is
                null, the packet is considered empty.

                \param[in] dt MPI datatype
                \param[in] p Pointer to the data of type value_type
             */
            packet(MPI_Datatype const &dt, value_type *p) : mpidt(dt), ptr(p), send_r(), recv_r() {}

            /** Function to check if the packet is actually associated with
                some data or not. If not the MPI send or recv is not
                issued. It is important to know that if on the other side the
                correspondig dual operation is issued, unexpected results
                would show up.

                \return true if the pointer in the packet is not nullptr
             */
            bool full() const { return ptr != nullptr; }

          private:
            friend struct all_to_all<value_type>;
        };

        /** This all to all class is explicitly designed to be a light
            wrapper around MPI. The idea is that each process has a vector
            of things to send to each other process, and a vector of data to
            be received from each other process. The data to be send or
            received is one element of a certain MPI_Datatype that is part
            of the array describing what to send. The pointer to the
            MPI_Datatype element is also part of the information in the
            array elements.

            \tparam vtype The type of the elements pointed by the pointer do the data to be sent or received.
         */
        template <typename vtype>
        struct all_to_all {
            typedef vtype value_type;

            const MPI_Comm a2a_comm;

            /** Array of packets defining what has to be sent to a given
                process. The element i of the array specify what to sent to
                process with ID i.
             */
            std::vector<packet<value_type>> to;

            /** Array of packets defining what has to be received from a given
                process. The element i of the array specify what to received
                from process with ID i.
             */
            std::vector<packet<value_type>> from;

            /** Constructor that takes the number of process (this work with
                MPI_COMM_WORLD, or GCL_WORLD communicators). The elements
                of the arrays are then initialized with empty packets (that
                would not trigger any communication call).

                \param[in] nprocs Number of processes in the MPI world
             */
            all_to_all(int nprocs) : a2a_comm(world()), to(nprocs), from(nprocs) {}

            /** Constructor that takes the number of process which takes a
                communicator to use during communication. The elements of the
                arrays are then initialized with empty packets (that would not
                trigger any communication call).

                \param[in] nprocs Number of processes in the MPI world
                \param[in] a2a_comm the MPI communicator to use
             */
            all_to_all(int nprocs, MPI_Comm a2a_comm) : a2a_comm(a2a_comm), to(nprocs), from(nprocs) {}

            /** This method prepare the pattern to be ready to execute
             */
            void setup() { MPI_Barrier(a2a_comm); }

            /** This method performs Irecevs from Isends to each process whose entry in the array of
                "from"s has a non null pointer.
             */
            void post_receives() {
                int tpid;
                MPI_Comm_rank(a2a_comm, &tpid);

                for (unsigned int i = 0; i < from.size(); ++i) {
                    if (from[i].full()) {
                        MPI_Irecv(from[i].ptr,
                            1,

                            from[i].mpidt,
                            i,
                            i,
                            a2a_comm,
                            &(from[i].recv_r));
                    }
                }
            }

            /** This method performs Isends to each process whose entry in the array of
                "to"s has a non null pointer.
             */
            void do_sends() {
                int tpid;
                MPI_Comm_rank(a2a_comm, &tpid);

                for (unsigned int i = 0; i < to.size(); ++i) {
                    if (to[i].full()) {
                        MPI_Isend(to[i].ptr, 1, to[i].mpidt, i, tpid, a2a_comm, &(to[i].send_r));
                    }
                }
            }

            /** This method starts the data exchange by posting receives and
                performing Isends to each process whose entry in the array of
                "to"s has a non null pointer.
             */
            void start_exchange() {
                post_receives();
                do_sends();
            }

            /** This function is called after the start_exchange() function to
                wait for the data that is supposed to arrive to each process.
             */
            void wait() {
                MPI_Status status;
                for (unsigned int i = 0; i < from.size(); ++i) {
                    if (from[i].full()) {
                        MPI_Wait(&(from[i].recv_r), &status);
                    }
                }
            }
        };
    } // namespace gcl
} // namespace gridtools
