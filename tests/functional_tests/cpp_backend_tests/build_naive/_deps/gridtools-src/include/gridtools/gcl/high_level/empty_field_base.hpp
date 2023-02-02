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

#include "../../common/halo_descriptor.hpp"
#include "../low_level/data_types_mapping.hpp"
#include "numerics.hpp"

namespace gridtools {
    namespace gcl {
        namespace _impl {
            template <typename value_type>
            struct make_datatype_outin {
                static MPI_Datatype type() { return compute_type<value_type>().value; }

                template <typename arraytype, typename arraytype2>
                static std::pair<MPI_Datatype, bool> outside(arraytype const &halo, arraytype2 const &eta) {
                    const int d = halo.size();
                    std::vector<int> sizes(d), subsizes(d), starts(d);

                    int ssz = 1;
                    for (int i = 0; i < d; ++i) {
                        sizes[i] = halo[i].total_length();
                        subsizes[i] =
                            halo[i].loop_high_bound_outside(eta[i]) - halo[i].loop_low_bound_outside(eta[i]) + 1;
                        ssz *= subsizes[i];
                        starts[i] = halo[i].loop_low_bound_outside(eta[i]);
                    }

                    if (ssz == 0)
                        return {MPI_INT, false};

                    MPI_Datatype res;
                    MPI_Type_create_subarray(d,
                        &sizes[0],
                        &subsizes[0],
                        &starts[0],
                        MPI_ORDER_FORTRAN, // increasing strides
                        type(),
                        &res);
                    MPI_Type_commit(&res);
                    return {res, true};
                }

                template <typename arraytype, typename arraytype2>
                static std::pair<MPI_Datatype, bool> inside(arraytype const &halo, arraytype2 const &eta) {
                    const int d = halo.size();
                    std::vector<int> sizes(d), subsizes(d), starts(d);

                    int ssz = 1;
                    for (int i = 0; i < d; ++i) {
                        sizes[i] = halo[i].total_length();
                        subsizes[i] =
                            halo[i].loop_high_bound_inside(eta[i]) - halo[i].loop_low_bound_inside(eta[i]) + 1;
                        ssz *= subsizes[i];
                        starts[i] = halo[i].loop_low_bound_inside(eta[i]);
                    }

                    if (ssz == 0)
                        return {MPI_INT, false};

                    MPI_Datatype res;
                    MPI_Type_create_subarray(d,
                        &sizes[0],
                        &subsizes[0],
                        &starts[0],
                        MPI_ORDER_FORTRAN, // increasing strides
                        type(),
                        &res);
                    MPI_Type_commit(&res);
                    return {res, true};
                }
            };

            template <typename Array>
            int neigh_idx(Array const &tuple) {
                int idx = 0;
                for (std::size_t i = 0; i < tuple.size(); ++i) {
                    int prod = 1;
                    for (std::size_t j = 0; j < i; ++j) {
                        prod = prod * 3;
                    }
                    idx = idx + (tuple[i] + 1) * prod;
                }
                return idx;
            }

            template <int_t I>
            struct neigh_loop {
                template <typename F, typename array>
                void operator()(F &f, array &tuple) {
                    for (int i = -1; i <= 1; ++i) {
                        tuple[I - 1] = i;
                        neigh_loop<I - 1>()(f, tuple);
                    }
                }
            };

            template <>
            struct neigh_loop<0> {
                template <typename F, typename array>
                void operator()(F &f, array &tuple) {
                    f(tuple);
                }
            };
        } // namespace _impl

        template <typename DataType>
        class empty_field_base {
            using DIMS_t = std::integral_constant<int, 3>;

            typedef array<halo_descriptor, DIMS_t::value> HALO_t;

          public:
            array<halo_descriptor, DIMS_t::value> halos;
            typedef array<std::pair<MPI_Datatype, bool>, static_pow3(DIMS_t::value)> MPDT_t;
            MPDT_t MPDT_OUTSIDE;
            MPDT_t MPDT_INSIDE;

            /**
                Function to set the halo descriptor of the field descriptor

                \param[in] D index of the dimension to be set
                \param[in] minus Please see field_descriptor_no_dt, halo_descriptor or \link MULTI_DIM_ACCESS \endlink
               for details \param[in] plus Please see field_descriptor_no_dt, halo_descriptor or \link MULTI_DIM_ACCESS
               \endlink for details \param[in] begin Please see field_descriptor_no_dt, halo_descriptor or \link
               MULTI_DIM_ACCESS \endlink for details \param[in] end Please see field_descriptor_no_dt, halo_descriptor
               or \link MULTI_DIM_ACCESS \endlink for details \param[in] t_len Please see field_descriptor_no_dt,
               halo_descriptor or \link MULTI_DIM_ACCESS \endlink for details
            */
            void add_halo(int D, int minus, int plus, int begin, int end, int t_len) {
                halos[D] = halo_descriptor(minus, plus, begin, end, t_len);
            }

            void add_halo(int D, halo_descriptor const &halo) { halos[D] = halo; }

            void setup() {
                array<int, DIMS_t::value> tuple;
                _impl::neigh_loop<DIMS_t::value>()(
                    [&](auto const &tuple) {
                        int idx = _impl::neigh_idx(tuple);
                        MPDT_OUTSIDE[idx] = _impl::make_datatype_outin<DataType>::outside(halos, tuple);
                        MPDT_INSIDE[idx] = _impl::make_datatype_outin<DataType>::inside(halos, tuple);
                    },
                    tuple);
            }

            std::pair<MPI_Datatype, bool> mpdt_inside(array<int, DIMS_t::value> const &eta) const {
                return MPDT_INSIDE[_impl::neigh_idx(eta)];
            }

            std::pair<MPI_Datatype, bool> mpdt_outside(array<int, DIMS_t::value> const &eta) const {
                return MPDT_OUTSIDE[_impl::neigh_idx(eta)];
            }

            /**
                Return the number of elements (not bytes) that have to be sent to a the neighbor
                indicated as an argument. This is the product of the lengths as in
                \link MULTI_DIM_ACCESS \endlink

                \param[in] eta the eta parameter as indicated in \link MULTI_DIM_ACCESS \endlink
            */
            int send_buffer_size(array<int, DIMS_t::value> const &eta) const {
                int S = 1;
                for (int i = 0; i < DIMS_t::value; ++i) {
                    S *= halos[i].s_length(eta[i]);
                }
                return S;
            }

            /**
                Return the number of elements (not bytes) that be receiver from the the neighbor
                indicated as an argument. This is the product of the lengths as in
                \link MULTI_DIM_ACCESS \endlink

                \param[in] eta the eta parameter as indicated in \link MULTI_DIM_ACCESS \endlink
            */
            int recv_buffer_size(array<int, DIMS_t::value> const &eta) const {
                int S = 1;
                for (int i = 0; i < DIMS_t::value; ++i) {
                    S *= halos[i].r_length(eta[i]);
                }
                return S;
            }
        };
    } // namespace gcl
} // namespace gridtools
