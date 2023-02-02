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

#include <type_traits>

#include "../../../common/integral_constant.hpp"

namespace gridtools {
    namespace stencil {
        namespace icosahedral {
            struct cells : integral_constant<int, 2> {};
            struct edges : integral_constant<int, 3> {};
            struct vertices : integral_constant<int, 1> {};

            template <class>
            struct is_location_type : std::false_type {};

            template <>
            struct is_location_type<cells> : std::true_type {};
            template <>
            struct is_location_type<edges> : std::true_type {};
            template <>
            struct is_location_type<vertices> : std::true_type {};
        } // namespace icosahedral
    }     // namespace stencil
} // namespace gridtools
