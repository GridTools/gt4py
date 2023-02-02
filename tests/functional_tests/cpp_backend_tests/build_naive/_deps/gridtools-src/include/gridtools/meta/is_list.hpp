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
         *   list concept check.
         *
         *   Note: it is not the same as is_instantiation_of<list, T>.
         */
        template <class>
        struct is_list : std::false_type {};
        template <template <class...> class L, class... Ts>
        struct is_list<L<Ts...>> : std::true_type {};
    } // namespace meta
} // namespace gridtools
