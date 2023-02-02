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

#include "concat.hpp"
#include "curry_fun.hpp"
#include "macros.hpp"

namespace gridtools {
    /**
     *  Filter the list based of predicate
     */
    namespace meta {
        namespace lazy {
            template <template <class...> class, class...>
            struct filter;
        }
        GT_META_DELEGATE_TO_LAZY(filter, (template <class...> class F, class... Args), (F, Args...));

        namespace lazy {
            template <bool, template <class...> class L, class T>
            struct wrap_if_impl {
                using type = L<T>;
            };

            template <template <class...> class L, class T>
            struct wrap_if_impl<false, L, T> {
                using type = L<>;
            };

            template <template <class...> class Pred>
            struct filter<Pred> {
                using type = curry_fun<meta::filter, Pred>;
            };
            template <template <class...> class Pred, template <class...> class L, class... Ts>
            struct filter<Pred, L<Ts...>> : concat<L<>, typename wrap_if_impl<Pred<Ts>::type::value, L, Ts>::type...> {
            };
        } // namespace lazy
    }     // namespace meta
} // namespace gridtools
