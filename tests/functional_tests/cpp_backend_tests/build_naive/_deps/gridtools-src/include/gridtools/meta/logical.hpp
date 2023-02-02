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

#include <cstddef>
#include <type_traits>

#include "list.hpp"
#include "macros.hpp"
#include "rename.hpp"
#include "repeat.hpp"
#include "transform.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  C++17 drop-offs
         *
         *  Note on conjunction_fast and disjunction_fast are like std counter parts but:
         *    - short-circuiting is not implemented as required by C++17 standard
         *    - amortized complexity is O(1) because of it [in terms of the number of template instantiations].
         */
        template <class... Ts>
        using conjunction_fast = std::is_same<list<std::integral_constant<bool, Ts::value>...>,
            repeat_c<GT_SIZEOF_3_DOTS(Ts), list<std::true_type>>>;

        template <class... Ts>
        using disjunction_fast = std::negation<conjunction_fast<std::negation<Ts>...>>;

        /**
         *   all elements in lists are true
         */
        template <class List>
        struct all : lazy::rename<conjunction_fast, List>::type {};

        /**
         *   some elements in lists are true
         */
        template <class List>
        struct any : lazy::rename<disjunction_fast, List>::type {};

        /**
         *  All elements satisfy predicate
         */
        template <template <class...> class Pred, class List>
        using all_of = all<transform<Pred, List>>;

        /**
         *  Some element satisfy predicate
         */
        template <template <class...> class Pred, class List>
        using any_of = any<transform<Pred, List>>;

        template <class...>
        struct are_same;

        template <>
        struct are_same<> : std::true_type {};

        template <class T, class... Ts>
        struct are_same<T, Ts...> : std::is_same<list<Ts...>, repeat_c<GT_SIZEOF_3_DOTS(Ts), list<T>>> {};

        template <class List>
        using all_are_same = rename<are_same, List>;
    } // namespace meta
} // namespace gridtools
