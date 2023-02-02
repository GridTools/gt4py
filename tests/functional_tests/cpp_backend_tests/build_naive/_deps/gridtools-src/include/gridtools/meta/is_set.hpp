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

#include "dedup.hpp"
#include "id.hpp"
#include "internal/inherit.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   True if the template parameter is type list which elements are all different
         */
        template <class>
        struct is_set : std::false_type {};

        template <template <class...> class L, class... Ts>
        struct is_set<L<Ts...>> : std::is_same<L<Ts...>, dedup<L<Ts...>>> {};

        /**
         *   is_set_fast evaluates to std::true_type if the parameter is a set.
         *   If parameter is not a type list, predicate evaluates to std::false_type.
         *   Compilation fails if the parameter is a type list with duplicated elements.
         *
         *   Its OK to use this predicate in static asserts and not OK in sfinae enablers.
         */
        template <class, class = void>
        struct is_set_fast : std::false_type {};

        template <template <class...> class L, class... Ts>
        struct is_set_fast<L<Ts...>, std::void_t<decltype(internal::inherit<lazy::id<Ts>...>{})>> : std::true_type {};
    } // namespace meta
} // namespace gridtools
