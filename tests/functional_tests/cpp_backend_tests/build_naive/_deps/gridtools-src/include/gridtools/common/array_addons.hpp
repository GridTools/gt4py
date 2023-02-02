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

#include <iostream>

#include "array.hpp"

namespace gridtools {
    /** \addtogroup common
        @{
     */

    /** \addtogroup array
        @{
    */

    template <typename T, size_t D>
    std::ostream &operator<<(std::ostream &s, array<T, D> const &a) {
        s << " {  ";
        if (D != 0) {
            for (size_t i = 0; i < D - 1; ++i)
                s << a[i] << ", ";

            s << a[D - 1];
        }
        return s << "  } ";
    }
    /** @} */
    /** @} */

} // namespace gridtools

/** \addtogroup common Common Shared Utilities
    @{
*/

/** \addtogroup array
    @{
*/

/** @} */
/** @} */
