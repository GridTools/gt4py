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

/**
   @file
   @brief On all boundary the values are copied from the last data field to the first. Minimum 2 fields.
*/

namespace gridtools {
    namespace boundaries {
        /** \ingroup Boundary-Conditions
         * @{
         */

        struct copy_boundary {

            /**   @brief On all boundary the values are copied from the last data field to the first. Minimum 2 fields.
             */
            template <typename Direction, typename DataField0, typename DataField1>
            GT_FUNCTION void operator()(
                Direction, DataField0 &data_field0, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
                data_field0(i, j, k) = data_field1(i, j, k);
            }

            template <typename Direction, typename DataField0, typename DataField1, typename DataField2>
            GT_FUNCTION void operator()(Direction,
                DataField0 &data_field0,
                DataField1 &data_field1,
                DataField2 const &data_field2,
                uint_t i,
                uint_t j,
                uint_t k) const {
                data_field0(i, j, k) = data_field2(i, j, k);
                data_field1(i, j, k) = data_field2(i, j, k);
            }
        };
        /** @} */
    } // namespace boundaries
} // namespace gridtools
