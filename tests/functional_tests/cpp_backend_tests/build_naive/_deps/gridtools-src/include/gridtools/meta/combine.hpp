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

#include "curry_fun.hpp"
#include "drop_front.hpp"
#include "length.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Applies binary function to the elements of the list.
         *
         *   For example:
         *     combine<f>::apply<list<t1, t2, t3, t4, t5, t6, t7>> === f<f<f<t1, t2>, f<t3, f4>>, f<f<t5, t6>, t7>>
         *
         *   Complexity is amortized O(N), the depth of template instantiation is O(log(N))
         */
        namespace lazy {
            template <template <class...> class, class...>
            struct combine;
        }
        GT_META_DELEGATE_TO_LAZY(combine, (template <class...> class F, class... Args), (F, Args...));

        namespace lazy {
            template <template <class...> class F, class List, std::size_t N>
            struct combine_impl {
                static_assert(N > 0, "N in combine_impl<F, List, N> must be positive");
                static constexpr std::size_t m = N / 2;
                using type = F<typename combine_impl<F, List, m>::type,
                    typename combine_impl<F, typename drop_front_c<m, List>::type, N - m>::type>;
            };
            template <template <class...> class F, class List>
            struct combine_impl<F, List, 0> {
                using type = F<>;
            };
            template <template <class...> class F, template <class...> class L, class T, class... Ts>
            struct combine_impl<F, L<T, Ts...>, 1> {
                using type = T;
            };
            template <template <class...> class F, template <class...> class L, class T1, class T2, class... Ts>
            struct combine_impl<F, L<T1, T2, Ts...>, 2> {
                using type = F<T1, T2>;
            };
            template <template <class...> class F,
                template <class...>
                class L,
                class T1,
                class T2,
                class T3,
                class... Ts>
            struct combine_impl<F, L<T1, T2, T3, Ts...>, 3> {
                using type = F<T1, F<T2, T3>>;
            };
            template <template <class...> class F,
                template <class...>
                class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts>
            struct combine_impl<F, L<T1, T2, T3, T4, Ts...>, 4> {
                using type = F<F<T1, T2>, F<T3, T4>>;
            };
            template <template <class...> class F>
            struct combine<F> {
                using type = curry_fun<meta::combine, F>;
            };
            template <template <class...> class F, class List>
            struct combine<F, List> : combine_impl<F, List, length<List>::value> {};
        } // namespace lazy
    }     // namespace meta
} // namespace gridtools
