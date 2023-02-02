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
            struct first;
            template <template <class...> class L, class T, class... Ts>
            struct first<L<T, Ts...>> {
                using type = T;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(first, (class List), (List));
    } // namespace meta
} // namespace gridtools
