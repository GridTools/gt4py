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

#include "macros.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <class>
            struct pop_front;

            template <template <class...> class L, class T, class... Ts>
            struct pop_front<L<T, Ts...>> {
                using type = L<Ts...>;
            };
            template <template <class...> class L>
            struct pop_front<L<>> {
                using type = L<>;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(pop_front, class List, List);
    } // namespace meta
} // namespace gridtools
