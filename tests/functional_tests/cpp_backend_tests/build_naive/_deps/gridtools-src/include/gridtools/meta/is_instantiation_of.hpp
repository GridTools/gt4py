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

#include "curry_fun.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Check if L is a ctor of List
         */
        template <template <class...> class L, class... Args>
        struct is_instantiation_of;
        template <template <class...> class L>
        struct is_instantiation_of<L> : curry_fun<is_instantiation_of, L> {};
        template <template <class...> class L, class T>
        struct is_instantiation_of<L, T> : std::false_type {};
        template <template <class...> class L, class... Ts>
        struct is_instantiation_of<L, L<Ts...>> : std::true_type {};
    } // namespace meta
} // namespace gridtools
