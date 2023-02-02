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
#include "macros.hpp"
#include "push_front.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <template <class...> class Pred, template <class...> class F, class List>
            struct group;

            template <template <class...> class Pred, template <class...> class F, template <class...> class L>
            struct group<Pred, F, L<>> {
                using type = L<>;
            };

            template <template <class...> class Pred, template <class...> class F, template <class...> class L, class T>
            struct group<Pred, F, L<T>> {
                using type = L<F<T>>;
            };

            template <template <class...> class Pred, class List, class Group>
            struct continue_grouping_impl : std::false_type {};

            template <template <class...> class Pred, template <class...> class L, class T, class... Ts, class... Us>
            struct continue_grouping_impl<Pred, L<T, Ts...>, meta::list<Us...>> : Pred<T, Us...> {};

            template <template <class...> class Pred,
                template <class...>
                class F,
                class List,
                class Group,
                bool = continue_grouping_impl<Pred, List, Group>::value>
            struct group_helper : push_front<typename group<Pred, F, List>::type, meta::rename<F, Group>> {};

            template <template <class...> class Pred,
                template <class...>
                class F,
                template <class...>
                class L,
                class T,
                class... Ts,
                class Group>
            struct group_helper<Pred, F, L<T, Ts...>, Group, true>
                : group_helper<Pred, F, L<Ts...>, typename push_back<Group, T>::type> {};

            template <template <class...> class Pred,
                template <class...>
                class F,
                template <class...>
                class L,
                class T,
                class... Ts>
            struct group<Pred, F, L<T, Ts...>> : group_helper<Pred, F, L<Ts...>, list<T>> {};
        } // namespace lazy

        /**
         *   Groups sequential elements of the input `List` that satisfy the given predicate `Pred` being applied to
         *   the group by applying function `F` to the group.
         *   Example: Say we have the list of non unique types. We want to convert input in a list of pairs where
         *   the pair consists of the the type and the number of sequential as an integral constant:
         *   ```
         *   template <class... Ts>
         *   using converter = meta::list<meta::first<meta::list<Ts...>, integral_constant<int, sizeof...(Ts)>>;
         *
         *   using src_t = ...
         *
         *   using result_t = meta::group<meta::are_same, converter, src_t>;
         *   ```
         */
        GT_META_DELEGATE_TO_LAZY(
            group, (template <class...> class Pred, template <class...> class F, class List), (Pred, F, List));
    } // namespace meta
} // namespace gridtools
