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

#include "../../common/defs.hpp"
#include "../../common/integral_constant.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            struct parallel {};
            struct forward {};
            struct backward {};

            template <class T>
            using is_parallel = std::is_same<T, parallel>;

            template <class T>
            using is_forward = std::is_same<T, forward>;

            template <class T>
            using is_backward = std::is_same<T, backward>;

            template <class T>
            constexpr integral_constant<int_t, is_backward<T>::value ? -1 : 1> step = {};
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
