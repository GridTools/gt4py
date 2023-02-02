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

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/halo_descriptor.hpp"
#include "direction.hpp"
#include "predicate.hpp"

/**
@file
@brief  definition of the functions which apply the boundary conditions (arbitrary functions having as argument the
direation, an arbitrary number of data fields, and the coordinates ID)
*/
namespace gridtools {
    namespace boundaries {
        /** \ingroup Boundary-Conditions
         * @{
         */

        template <typename BoundaryFunction,
            typename Predicate = default_predicate,
            typename HaloDescriptors = array<halo_descriptor, 3u>>
        struct boundary_apply {
          private:
            HaloDescriptors halo_descriptors;
            BoundaryFunction const boundary_function;
            Predicate predicate;

            /** @brief loops on the halo region defined by the HaloDescriptor member parameter, and evaluates the
               boundary_function in the specified direction, in the specified halo node.
                this macro expands to n definitions of the function loop, taking a number of arguments ranging from 0 to
               n (DataField0, Datafield1, DataField2, ...)*/
            template <typename Direction, typename... DataField>
            void loop(DataField &...data_field) const {
                const int_t i_low = halo_descriptors[0].loop_low_bound_outside(Direction::i);
                const int_t i_high = halo_descriptors[0].loop_high_bound_outside(Direction::i);
                const int_t j_low = halo_descriptors[1].loop_low_bound_outside(Direction::j);
                const int_t j_high = halo_descriptors[1].loop_high_bound_outside(Direction::j);
                const int_t k_low = halo_descriptors[2].loop_low_bound_outside(Direction::k);
                const int_t k_high = halo_descriptors[2].loop_high_bound_outside(Direction::k);

#pragma omp parallel for simd collapse(3)
                for (int_t j = j_low; j <= j_high; ++j)
                    for (int_t k = k_low; k <= k_high; ++k)
                        for (int_t i = i_low; i <= i_high; ++i)
                            boundary_function(Direction(), data_field..., i, j, k);
            }

          public:
            boundary_apply(HaloDescriptors const &hd, Predicate predicate = Predicate())
                : halo_descriptors(hd), boundary_function(BoundaryFunction()), predicate(predicate) {}

            boundary_apply(HaloDescriptors const &hd, BoundaryFunction const &bf, Predicate predicate = Predicate())
                : halo_descriptors(hd), boundary_function(bf), predicate(predicate) {}

            /**
               @brief applies the boundary conditions looping on the halo region defined by the member parameter, in all
            possible directions.
            this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n
            (DataField0, Datafield1, DataField2, ...)

            */
            template <typename... DataFieldViews>
            void apply(DataFieldViews const &...data_field_views) const {

                if (predicate(direction<minus_, minus_, minus_>()))
                    this->loop<direction<minus_, minus_, minus_>>(data_field_views...);
                if (predicate(direction<minus_, minus_, zero_>()))
                    this->loop<direction<minus_, minus_, zero_>>(data_field_views...);
                if (predicate(direction<minus_, minus_, plus_>()))
                    this->loop<direction<minus_, minus_, plus_>>(data_field_views...);

                if (predicate(direction<minus_, zero_, minus_>()))
                    this->loop<direction<minus_, zero_, minus_>>(data_field_views...);
                if (predicate(direction<minus_, zero_, zero_>()))
                    this->loop<direction<minus_, zero_, zero_>>(data_field_views...);
                if (predicate(direction<minus_, zero_, plus_>()))
                    this->loop<direction<minus_, zero_, plus_>>(data_field_views...);

                if (predicate(direction<minus_, plus_, minus_>()))
                    this->loop<direction<minus_, plus_, minus_>>(data_field_views...);
                if (predicate(direction<minus_, plus_, zero_>()))
                    this->loop<direction<minus_, plus_, zero_>>(data_field_views...);
                if (predicate(direction<minus_, plus_, plus_>()))
                    this->loop<direction<minus_, plus_, plus_>>(data_field_views...);

                if (predicate(direction<zero_, minus_, minus_>()))
                    this->loop<direction<zero_, minus_, minus_>>(data_field_views...);
                if (predicate(direction<zero_, minus_, zero_>()))
                    this->loop<direction<zero_, minus_, zero_>>(data_field_views...);
                if (predicate(direction<zero_, minus_, plus_>()))
                    this->loop<direction<zero_, minus_, plus_>>(data_field_views...);

                if (predicate(direction<zero_, zero_, minus_>()))
                    this->loop<direction<zero_, zero_, minus_>>(data_field_views...);
                if (predicate(direction<zero_, zero_, plus_>()))
                    this->loop<direction<zero_, zero_, plus_>>(data_field_views...);

                if (predicate(direction<zero_, plus_, minus_>()))
                    this->loop<direction<zero_, plus_, minus_>>(data_field_views...);
                if (predicate(direction<zero_, plus_, zero_>()))
                    this->loop<direction<zero_, plus_, zero_>>(data_field_views...);
                if (predicate(direction<zero_, plus_, plus_>()))
                    this->loop<direction<zero_, plus_, plus_>>(data_field_views...);

                if (predicate(direction<plus_, minus_, minus_>()))
                    this->loop<direction<plus_, minus_, minus_>>(data_field_views...);
                if (predicate(direction<plus_, minus_, zero_>()))
                    this->loop<direction<plus_, minus_, zero_>>(data_field_views...);
                if (predicate(direction<plus_, minus_, plus_>()))
                    this->loop<direction<plus_, minus_, plus_>>(data_field_views...);

                if (predicate(direction<plus_, zero_, minus_>()))
                    this->loop<direction<plus_, zero_, minus_>>(data_field_views...);
                if (predicate(direction<plus_, zero_, zero_>()))
                    this->loop<direction<plus_, zero_, zero_>>(data_field_views...);
                if (predicate(direction<plus_, zero_, plus_>()))
                    this->loop<direction<plus_, zero_, plus_>>(data_field_views...);

                if (predicate(direction<plus_, plus_, minus_>()))
                    this->loop<direction<plus_, plus_, minus_>>(data_field_views...);
                if (predicate(direction<plus_, plus_, zero_>()))
                    this->loop<direction<plus_, plus_, zero_>>(data_field_views...);
                if (predicate(direction<plus_, plus_, plus_>()))
                    this->loop<direction<plus_, plus_, plus_>>(data_field_views...);

                // apply(data_field_views ...);
            }

          private:
            /** fixing compilation */
            void apply() const {}
        };
    } // namespace boundaries
    /** @} */
} // namespace gridtools
