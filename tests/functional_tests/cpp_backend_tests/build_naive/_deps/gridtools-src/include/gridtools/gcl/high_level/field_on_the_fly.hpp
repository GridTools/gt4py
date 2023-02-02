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

#include "../../common/array.hpp"
#include "../../common/halo_descriptor.hpp"
#include "../../common/layout_map.hpp"

namespace gridtools {
    namespace gcl {
        /**
           Struct that contains the information for an array with halo. It
           construct all necessary information to execute halo_exchange_generic

           \tparam DataType Type of the elements stored in the array
           \tparam DIMS Number of dimensions of the array
           \tparam layoutmap Specification of the layout map of the data (as in halo_exchange_dynamic)
         */
        template <typename DataType, typename _t_layoutmap, template <typename> class Traits>
        struct field_on_the_fly : public Traits<DataType>::base_field {
            // This is necessary since the internals of gcl use "increasing stride order" instead of "decreasing stride
            // order"
            using inner_layoutmap = reverse_map<_t_layoutmap>;
            typedef _t_layoutmap outer_layoutmap;
            static const int DIMS = Traits<DataType>::I;

            typedef typename Traits<DataType>::base_field base_type;

            typedef field_on_the_fly<DataType, _t_layoutmap, Traits> this_type;

            typedef DataType value_type;

            mutable DataType *ptr;

            field_on_the_fly() = default;

            template <typename T1>
            field_on_the_fly<T1, _t_layoutmap, Traits> &retarget() {
                void *tmp = this;
                return *(reinterpret_cast<field_on_the_fly<T1, _t_layoutmap, Traits> *>(tmp));
            }

            template <typename T1>
            field_on_the_fly<T1, _t_layoutmap, Traits> copy() const {
                const void *tmp = this;
                return *(reinterpret_cast<const field_on_the_fly<T1, _t_layoutmap, Traits> *>(tmp));
            }

            void set_pointer(DataType *pointer) { ptr = pointer; }

            DataType *get_pointer() const { return ptr; }

            /**
               Constructor that takes an gridtools::array of halo descriptors. The order
               of the elements are the logical order in which the user sees the
               dimensions. Layout map is used to permute the entries in the proper
               way.

               \param p Pointer to the array containing the data
               \param halos Array (gridtools::array) of array halos
             */
            field_on_the_fly(DataType *p, array<halo_descriptor, DIMS> const &halos) : ptr(p) {
                for (int i = 0; i < DIMS; ++i) {
                    base_type::add_halo(inner_layoutmap::at(i),
                        halos[i].minus(),
                        halos[i].plus(),
                        halos[i].begin(),
                        halos[i].end(),
                        halos[i].total_length());
                }

                base_type::setup();
            }
            /**
               Method to explicitly create a field_on_the_fly. It takes an gridtools::array
               of halo descriptors. The order of the elements are the logical order in
               which the user sees the dimensions. Layout map is used to permute the
               entries in the proper way.

               \param p Pointer to the array containing the data
               \param halos Array (gridtools::array) of array halos
             */
            void create(DataType *p, array<halo_descriptor, DIMS> const &halos) {
                ptr = p;
                for (int i = 0; i < DIMS; ++i) {
                    base_type::add_halo(inner_layoutmap::at(i),
                        halos[i].minus(),
                        halos[i].plus(),
                        halos[i].begin(),
                        halos[i].end(),
                        halos[i].total_length());
                }

                base_type::setup();
            }

            const DataType *the_pointer() const { return ptr; }
        };
    } // namespace gcl
} // namespace gridtools
