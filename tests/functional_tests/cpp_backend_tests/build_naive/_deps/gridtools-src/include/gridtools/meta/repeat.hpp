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

#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Produce a list of N identical elements
         */
        namespace lazy {
            template <class List, bool Rem>
            struct repeat_impl_expand;

            template <template <class...> class L, class... Ts>
            struct repeat_impl_expand<L<Ts...>, false> {
                using type = L<Ts..., Ts...>;
            };

            template <template <class...> class L, class T, class... Ts>
            struct repeat_impl_expand<L<T, Ts...>, true> {
                using type = L<T, T, T, Ts..., Ts...>;
            };

            template <std::size_t N, class L>
            struct repeat_c {
                using type = typename repeat_impl_expand<typename repeat_c<N / 2, L>::type, N % 2>::type;
            };

            template <template <class...> class L, class... Ts>
            struct repeat_c<0, L<Ts...>> {
                using type = L<>;
            };

            template <template <class...> class L, class... Ts>
            struct repeat_c<1, L<Ts...>> {
                using type = L<Ts...>;
            };

            template <class N, class L>
            using repeat = repeat_c<N::value, L>;
        } // namespace lazy
        template <std::size_t N, class T>
        using repeat_c = typename lazy::repeat_c<N, T>::type;
        template <class N, class T>
        using repeat = typename lazy::repeat_c<N::value, T>::type;
    } // namespace meta
} // namespace gridtools
