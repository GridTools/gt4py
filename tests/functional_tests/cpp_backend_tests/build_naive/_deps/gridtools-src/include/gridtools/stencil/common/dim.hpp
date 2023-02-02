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

#include "../../common/integral_constant.hpp"

namespace gridtools {
    namespace stencil {
        namespace dim {
            using i = integral_constant<int, 0>;
            using j = integral_constant<int, 1>;
            using k = integral_constant<int, 2>;
            using c = integral_constant<int, 3>;

            struct thread;
        } // namespace dim
    }     // namespace stencil
} // namespace gridtools
