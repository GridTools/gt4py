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

#include "list.hpp"
#include "macros.hpp"

#include "fold.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Concatenate lists
         */
        namespace lazy {
            template <class...>
            struct concat;
        }
        GT_META_DELEGATE_TO_LAZY(concat, class... Lists, Lists...);
        namespace lazy {
            template <>
            struct concat<> : list<> {};
            template <template <class...> class L, class... Ts>
            struct concat<L<Ts...>> {
                using type = L<Ts...>;
            };
            template <template <class...> class L1, class... T1s, template <class...> class L2, class... T2s>
            struct concat<L1<T1s...>, L2<T2s...>> {
                using type = L1<T1s..., T2s...>;
            };

            template <class List, class... Lists>
            struct concat<List, Lists...> : foldl<meta::concat, List, list<Lists...>> {};
        } // namespace lazy
    }     // namespace meta
} // namespace gridtools
