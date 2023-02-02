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

#include "id.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "repeat.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Drop N elements from the front of the list
         *
         *  Complexity is amortized O(1).
         */
        namespace lazy {
            template <class SomeList, class List>
            class drop_front_impl;
            template <class... Us, template <class...> class L, class... Ts>
            class drop_front_impl<list<Us...>, L<Ts...>> {
                template <class... Vs>
                static L<Vs...> select(Us *..., id<Vs> *...);

              public:
                using type = decltype(select(((id<Ts> *)0)...));
            };

            template <class N, class List>
            using drop_front = drop_front_impl<typename repeat_c<N::value, list<void>>::type, List>;

            template <std::size_t N, class List>
            using drop_front_c = drop_front_impl<typename repeat_c<N, list<void>>::type, List>;
        } // namespace lazy
        template <std::size_t N, class List>
        using drop_front_c = typename lazy::drop_front_impl<typename repeat_c<N, list<void>>::type, List>::type;
        template <class N, class List>
        using drop_front = typename lazy::drop_front_impl<typename repeat_c<N::value, list<void>>::type, List>::type;
    } // namespace meta
} // namespace gridtools
