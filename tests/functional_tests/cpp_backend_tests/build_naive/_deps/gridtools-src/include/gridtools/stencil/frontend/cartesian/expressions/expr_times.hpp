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

#include "../../../../common/host_device.hpp"
#include "expr_base.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace expressions {
                struct times_f {
                    template <class Lhs, class Rhs>
                    GT_FUNCTION constexpr auto operator()(Lhs const &lhs, Rhs const &rhs) const {
                        return lhs * rhs;
                    }
                };

                template <class Lhs, class Rhs>
                GT_FUNCTION constexpr auto operator*(Lhs lhs, Rhs rhs) -> decltype(make_expr(times_f(), lhs, rhs)) {
                    return make_expr(times_f(), lhs, rhs);
                }
            } // namespace expressions
        }     // namespace cartesian
    }         // namespace stencil
} // namespace gridtools
