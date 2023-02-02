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

namespace gridtools {
    namespace meta {
        /**
         *  Check if the class has inner `type`
         */
        template <class, class = void>
        struct has_type : std::false_type {};
        template <class T>
        struct has_type<T, std::void_t<typename T::type>> : std::true_type {};
    } // namespace meta
} // namespace gridtools
