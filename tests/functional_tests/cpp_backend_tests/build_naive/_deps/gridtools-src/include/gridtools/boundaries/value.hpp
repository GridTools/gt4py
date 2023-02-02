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

#include "../common/defs.hpp"
#include "../common/host_device.hpp"

namespace gridtools {
    namespace boundaries {
        /** \ingroup Boundary-Conditions
         * @{
         */

        /**
           On all boundary the values ares set to a fixed value,
           which is zero for basic data types.

           \tparam T The type of the value to be assigned
         */
        template <typename T>
        struct value_boundary {

            /**
               Constructor that assigns the value to set the boundaries
             */
            value_boundary(T const &a) : value(a) {}

            /**
               Constructor that assigns the default constructed value to set the boundaries
             */
            value_boundary() : value() {}

            template <typename Direction, typename DataField0>
            GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
                data_field0(i, j, k) = value;
            }

            template <typename Direction, typename DataField0, typename DataField1>
            GT_FUNCTION void operator()(
                Direction, DataField0 &data_field0, DataField1 &data_field1, uint_t i, uint_t j, uint_t k) const {
                data_field0(i, j, k) = value;
                data_field1(i, j, k) = value;
            }

            template <typename Direction, typename DataField0, typename DataField1, typename DataField2>
            GT_FUNCTION void operator()(Direction,
                DataField0 &data_field0,
                DataField1 &data_field1,
                DataField2 &data_field2,
                uint_t i,
                uint_t j,
                uint_t k) const {
                data_field0(i, j, k) = value;
                data_field1(i, j, k) = value;
                data_field2(i, j, k) = value;
            }

          private:
            T value;
        };
        /** @} */
    } // namespace boundaries
} // namespace gridtools
