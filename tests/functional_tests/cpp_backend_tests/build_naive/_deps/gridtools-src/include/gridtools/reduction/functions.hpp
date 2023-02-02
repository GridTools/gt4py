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

#include "../common/host_device.hpp"

namespace gridtools {
    namespace reduction {
        struct plus {
            template <class T>
            GT_FUNCTION auto operator()(T const &x, T const &y) const {
                return x + y;
            }
        };
        struct mul {
            template <class T>
            GT_FUNCTION auto operator()(T const &x, T const &y) const {
                return x * y;
            }
        };
        struct min {
            template <class T>
            GT_FUNCTION auto operator()(T const &x, T const &y) const {
                return x < y ? x : y;
            }
        };
        struct max {
            template <class T>
            GT_FUNCTION auto operator()(T const &x, T const &y) const {
                return x > y ? x : y;
            }
        };
        struct bitwise_and {
            template <class T>
            GT_FUNCTION auto operator()(T const &x, T const &y) const {
                return x & y;
            }
        };
        struct bitwise_or {
            template <class T>
            GT_FUNCTION auto operator()(T const &x, T const &y) const {
                return x | y;
            }
        };
        struct bitwise_xor {
            template <class T>
            GT_FUNCTION auto operator()(T const &x, T const &y) const {
                return x ^ y;
            }
        };
    } // namespace reduction
} // namespace gridtools
