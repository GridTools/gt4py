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

#include "curry_fun.hpp"
#include "fold.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "push_back.hpp"
#include "rename.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Transform `Lists` by applying `F` element wise.
         *
         *   I.e the first element of resulting list would be `F<first_from_l0, first_froml1, ...>`;
         *   the second would be `F<second_from_l0, ...>` and so on.
         *
         *   For N lists M elements each complexity is O(N). I.e for one list it is O(1).
         */
        namespace lazy {
            template <template <class...> class, class...>
            struct transform;
        }
        GT_META_DELEGATE_TO_LAZY(transform, (template <class...> class F, class... Args), (F, Args...));

        namespace lazy {
            template <template <class...> class F>
            struct transform<F> {
                using type = curry_fun<meta::transform, F>;
            };
            template <template <class...> class F, template <class...> class L, class... Ts>
            struct transform<F, L<Ts...>> {
                using type = L<F<Ts>...>;
            };
            template <template <class...> class F,
                template <class...>
                class L1,
                class... T1s,
                template <class...>
                class L2,
                class... T2s>
            struct transform<F, L1<T1s...>, L2<T2s...>> {
                using type = L1<F<T1s, T2s>...>;
            };

            /**
             *   Takes `2D array` of types (i.e. list of lists where inner lists are the same length) and do
             *   trasposition. Example:
             *   a<b<void, void*, void**>, b<int, int*, int**>> => b<a<void, int>, a<void*, int*>, a<void**, int**>>
             */
            template <class>
            struct transpose;
            template <template <class...> class L>
            struct transpose<L<>> {
                using type = list<>;
            };
            template <template <class...> class Outer, template <class...> class Inner, class... Ts, class... Inners>
            struct transpose<Outer<Inner<Ts...>, Inners...>>
                : foldl<transform<meta::push_back>::type::apply, Inner<Outer<Ts>...>, list<Inners...>> {};

            // transform, generic version
            template <template <class...> class F, class List, class... Lists>
            struct transform<F, List, Lists...>
                : transform<rename<F>::type::template apply, typename transpose<list<List, Lists...>>::type> {};
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(transpose, class List, List);
    } // namespace meta
} // namespace gridtools
