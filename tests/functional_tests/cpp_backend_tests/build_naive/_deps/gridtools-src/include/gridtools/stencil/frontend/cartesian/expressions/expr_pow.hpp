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

#include "../../../../common/gt_math.hpp"
#include "../../../../common/host_device.hpp"
#include "expr_base.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace expressions {
                template <int I>
                struct pow_f {
                    template <class Arg>
                    GT_FUNCTION constexpr auto operator()(Arg const &arg) const {
                        return gt_pow<I>::template apply(arg);
                    }
                };

                template <int I, class Arg>
                GT_FUNCTION constexpr auto pow(Arg arg) -> decltype(make_expr(pow_f<I>(), Arg())) {
                    return make_expr(pow_f<I>(), arg);
                }
            } // namespace expressions
        }     // namespace cartesian
    }         // namespace stencil
} // namespace gridtools
