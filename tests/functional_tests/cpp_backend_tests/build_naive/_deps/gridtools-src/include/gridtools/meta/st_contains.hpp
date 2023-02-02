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

#include "id.hpp"
#include "internal/inherit.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   true_type if Set contains T
         *
         *   "st_" prefix stands for set
         *
         *  @pre All elements of Set are unique.
         *
         *  Complexity is O(1)
         */
        template <class Set, class T>
        struct st_contains : std::false_type {};
        template <template <class...> class L, class... Ts, class T>
        struct st_contains<L<Ts...>, T> : std::is_base_of<lazy::id<T>, internal::inherit<lazy::id<Ts>...>> {};
    } // namespace meta
} // namespace gridtools
