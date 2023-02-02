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

namespace gridtools {
    namespace gcl {
        template <typename v_type>
        struct compute_type {
            // This is called when no other type is found.
            // In this case the type is considered POD
            // No static value available here
            MPI_Datatype value;

            compute_type() {
                MPI_Type_contiguous(sizeof(v_type), MPI_CHAR, &value);
                MPI_Type_commit(&value);
            }
        };

        template <>
        struct compute_type<int> {
            const MPI_Datatype value;
            compute_type() : value(MPI_INT) {}
        };

        template <>
        struct compute_type<char> {
            const MPI_Datatype value;
            compute_type() : value(MPI_CHAR) {}
        };

        template <>
        struct compute_type<float> {
            const MPI_Datatype value;
            compute_type() : value(MPI_FLOAT) {}
        };

        template <>
        struct compute_type<double> {
            const MPI_Datatype value;
            compute_type() : value(MPI_DOUBLE) {}
        };

        template <typename value_type>
        struct make_datatype {

          public:
            static MPI_Datatype type() { return compute_type<value_type>().value; }

            template <typename arraytype>
            static MPI_Datatype make(arraytype const &halo) {
                const int d = halo.size();
                std::vector<int> sizes(d), subsizes(d), starts(d);

                for (int i = 0; i < d; ++i) {
                    sizes[i] = halo[i].total_length();
                    subsizes[i] = halo[i].end() - halo[i].begin() + 1;
                    starts[i] = halo[i].begin();
                }

                MPI_Datatype res;
                MPI_Type_create_subarray(d,
                    &sizes[0],
                    &subsizes[0],
                    &starts[0],
                    MPI_ORDER_C, // decreasing strides
                    type(),
                    &res);
                MPI_Type_commit(&res);
                return res;
            }
        };
    } // namespace gcl
} // namespace gridtools
