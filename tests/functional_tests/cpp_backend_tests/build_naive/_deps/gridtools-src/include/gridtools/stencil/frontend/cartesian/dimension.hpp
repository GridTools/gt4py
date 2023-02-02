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

#include <cstddef>

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            /**
               @brief The following struct defines one specific component of a field
               It contains a direction (compile time constant, specifying the ID of the component),
               and a value (runtime value, which is storing the offset in the given direction).
            */
            template <std::size_t Coordinate>
            struct dimension {
                static_assert(Coordinate != 0, "The coordinate values passed to the accessor start from 1");

                GT_FUNCTION constexpr dimension(int_t value = 0) : value(value) {}
                int_t value;

                friend GT_FUNCTION constexpr dimension operator+(dimension, int_t offset) { return {offset}; }
                friend GT_FUNCTION constexpr dimension operator-(dimension, int_t offset) { return {-offset}; }
            };
        } // namespace cartesian
    }     // namespace stencil
} // namespace gridtools
