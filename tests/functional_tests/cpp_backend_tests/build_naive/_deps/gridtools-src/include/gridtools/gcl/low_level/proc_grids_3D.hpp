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
#include <type_traits>

#include <mpi.h>

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "boollist.hpp"

// This file needs to be changed

namespace gridtools {
    namespace gcl {
        /** \class MPI_3D_process_grid_t
         * Class that provides a representation of a 3D process grid given an MPI CART
         * It requires the MPI CART to be defined before the grid is created
         * \tparam CYCLIC is a template argument matching \ref boollist_concept to specify periodicities
         * \n
         * This is a process grid matching the \ref proc_grid_concept concept
         */
        template <int Ndims>
        struct MPI_3D_process_grid_t {

            /** number of dimensions
             */
            static const int ndims = Ndims;

            typedef boollist<ndims> period_type;

          private:
            MPI_Comm m_communicator; // Communicator that is associated with the MPI CART!
            period_type m_cyclic;
            int m_nprocs;
            array<int, ndims> m_dimensions;
            array<int, ndims> m_coordinates;

          public:
            MPI_3D_process_grid_t(MPI_3D_process_grid_t const &other)
                : m_cyclic(other.cyclic()), m_nprocs(other.m_nprocs) {

                MPI_Comm_dup(other.m_communicator, &m_communicator);

                for (uint_t i = 0; i < ndims; ++i) {
                    m_dimensions[i] = other.m_dimensions[i];
                    m_coordinates[i] = other.m_coordinates[i];
                }
            }

            /** Constructor that takes an MPI CART communicator, already configured, and use it to set up the process
               grid. \param c Object containing information about periodicities as defined in \ref boollist_concept
                \param comm MPI Communicator describing the MPI 3D computing grid
            */
            MPI_3D_process_grid_t(period_type const &c, MPI_Comm const &comm)
                : m_cyclic(c), m_nprocs(0), m_dimensions(), m_coordinates() {

                MPI_Comm_dup(comm, &m_communicator);

                int period[ndims];
                MPI_Cart_get(comm, ndims, &m_dimensions[0], period, &m_coordinates[0]);
                MPI_Comm_size(comm, &m_nprocs);
            }

            /** Constructor that takes an MPI CART communicator, already configured, and use it to set up the process
               grid. \param c Object containing information about periodicities as defined in \ref boollist_concept
                \param comm MPI Communicator describing the MPI 3D computing grid
                \param dims Array of dimensions of the processor grid
            */
            template <typename Array>
            MPI_3D_process_grid_t(period_type const &c, MPI_Comm const &comm, Array const &dims)
                : m_communicator(), m_cyclic(c), m_nprocs(0), m_dimensions(dims), m_coordinates() {
                MPI_Comm_size(comm, &m_nprocs);
                MPI_Dims_create(m_nprocs, dims.size(), &m_dimensions[0]);
                int period[3] = {1, 1, 1};
                MPI_Cart_create(comm, 3, &m_dimensions[0], period, false, &m_communicator);
                MPI_Cart_get(
                    m_communicator, ndims, &m_dimensions[0], period /*does not really care*/, &m_coordinates[0]);
            }

            ~MPI_3D_process_grid_t() { MPI_Comm_free(&m_communicator); }

            /**
               Returns communicator
            */
            MPI_Comm communicator() const { return m_communicator; }

            /** Returns in t_R and t_C the lenght of the dimensions of the process grid AS PRESCRIBED BY THE CONCEPT
                \param[out] t_R Number of elements in first dimension
                \param[out] t_C Number of elements in second dimension
                \param[out] t_S Number of elements in third dimension
            */
            void dims(int &t_R, int &t_C, int &t_S) const {
                static_assert(ndims == 3, "this interface supposes ndims=3");
                t_R = m_dimensions[0];
                t_C = m_dimensions[1];
                t_S = m_dimensions[2];
            }

            /** Returns the dimensions in an array of dimensions (at least of size 3)
                \tparam The array type
                \param array The array where to put the values
            */
            template <class Array>
            void fill_dims(Array &array) const {
                static_assert(ndims == 3, "this interface supposes ndims=3");
                array[0] = m_dimensions[0];
                array[1] = m_dimensions[1];
                array[2] = m_dimensions[2];
            }

            void dims(int &t_R, int &t_C) const {
                static_assert(ndims == 2, "this interface supposes ndims=2");
                t_R = m_dimensions[0];
                t_C = m_dimensions[1];
            }

            /** Returns the number of processors of the processor grid

                \return Number of processors
            */
            uint_t size() const {
                uint_t ret = m_dimensions[0];
                for (uint_t i = 1; i < ndims; ++i)
                    ret *= m_dimensions[i];
                return ret;
            }

            /** Returns in t_R and t_C the coordinates ot the caller process in the grid AS PRESCRIBED BY THE CONCEPT
                \param[out] t_R Coordinate in first dimension
                \param[out] t_C Coordinate in second dimension
                \param[out] t_S Coordinate in third dimension
            */
            void coords(int &t_R, int &t_C, int &t_S) const {
                static_assert(ndims == 3, "this interface supposes ndims=3");
                t_R = m_coordinates[0];
                t_C = m_coordinates[1];
                t_S = m_coordinates[2];
            }

            void coords(int &t_R, int &t_C) const {
                static_assert(ndims == 2, "this interface supposes ndims=2");
                t_R = m_coordinates[0];
                t_C = m_coordinates[1];
            }

            /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process
               AS PRESCRIBED BY THE CONCEPT \tparam I Relative coordinate in the first dimension \tparam J Relative
               coordinate in the second dimension \tparam K Relative coordinate in the third dimension \return The
               process ID of the required process
            */
            template <int I, int J, int K>
            int proc() const {
                // int coords[3]={I,J,K};
                return proc(I, J, K);
            }

            int pid() const {
                int rank;
                MPI_Comm_rank(m_communicator, &rank);
                return rank;
            }

            /** Returns the process ID of the process with relative coordinates (I,J) with respect to the caller process
               AS PRESCRIBED BY THE CONCEPT \param[in] I Relative coordinate in the first dimension \param[in] J
               Relative coordinate in the second dimension \param[in] K Relative coordinate in the third dimension
                \return The process ID of the required process
            */
            int proc(int I, int J, int K) const {
                int _coords[3];

                if (m_cyclic.value(0))
                    _coords[0] = (m_coordinates[0] + I + m_dimensions[0]) % m_dimensions[0];
                else {
                    _coords[0] = m_coordinates[0] + I;
                    if (_coords[0] < 0 || _coords[0] >= m_dimensions[0])
                        return -1;
                }

                if (m_cyclic.value(1))
                    _coords[1] = (m_coordinates[1] + J + m_dimensions[1]) % m_dimensions[1];
                else {
                    _coords[1] = m_coordinates[1] + J;
                    if (_coords[1] < 0 || _coords[1] >= m_dimensions[1])
                        return -1;
                }

                if (m_cyclic.value(2))
                    _coords[2] = (m_coordinates[2] + K + m_dimensions[2]) % m_dimensions[2];
                else {
                    _coords[2] = m_coordinates[2] + K;
                    if (_coords[2] < 0 || _coords[2] >= m_dimensions[2])
                        return -1;
                }

                int pid = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &pid);
                int res;
                MPI_Cart_rank(m_communicator, _coords, &res);
                return res;
            }

            GT_FUNCTION
            array<int, ndims> const &coordinates() const { return m_coordinates; }

            GT_FUNCTION
            array<int, ndims> const &dimensions() const { return m_dimensions; }

            /** Returns the process ID of the process with absolute coordinates specified by the input array of
               coordinates
                \param[in] crds gridtools::aray of coordinates of the processor of which the ID is needed

                \return The process ID of the required process
            */
            int abs_proc(array<int, ndims> const &crds) const {
                return proc(crds[0] - m_coordinates[0], crds[1] - m_coordinates[1], crds[2] - m_coordinates[2]);
            }

            auto ntasks() { return m_nprocs; }

            bool periodic(int index) const {
                assert(index < ndims);
                return m_cyclic.value(index);
            }

            array<bool, ndims> periodic() const {
                static_assert(period_type::m_size == ndims, "Dimensions not matching");
                return m_cyclic.value();
            }

            decltype(auto) cyclic() const { return m_cyclic; }

            auto coordinates(uint_t i) const { return m_coordinates[i]; }
            auto dimensions(uint_t i) const { return m_dimensions[i]; }
        };
    } // namespace gcl
} // namespace gridtools
