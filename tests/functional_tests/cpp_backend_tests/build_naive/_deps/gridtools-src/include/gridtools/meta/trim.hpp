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

#include "macros.hpp"
#include "reverse.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <template <class...> class Pred, class List>
            struct remove_prefix;

            template <template <class...> class Pred, template <class...> class L>
            struct remove_prefix<Pred, L<>> {
                using type = L<>;
            };

            template <template <class...> class Pred, template <class...> class L, class T, class... Ts>
            struct remove_prefix<Pred, L<T, Ts...>>
                : std::conditional<Pred<T>::value, typename remove_prefix<Pred, L<Ts...>>::type, L<T, Ts...>> {};

            template <template <class...> class Pred, class List>
            struct remove_suffix : reverse<typename remove_prefix<Pred, typename reverse<List>::type>::type> {};

            template <template <class...> class Pred, class List>
            using trim = remove_suffix<Pred, typename remove_prefix<Pred, List>::type>;
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(remove_prefix, (template <class...> class Pred, class List), (Pred, List));
        GT_META_DELEGATE_TO_LAZY(remove_suffix, (template <class...> class Pred, class List), (Pred, List));

        /**
         *   Remove the prefix and the suffix of the type list that satisfy the given predicate.
         *   I.e. all elements that satisfy the predicate are removed from the beginning and from the end but not
         *   removed in the middle.
         */
        template <template <class...> class Pred, class List>
        using trim = remove_suffix<Pred, remove_prefix<Pred, List>>;
    } // namespace meta
} // namespace gridtools
