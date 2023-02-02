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
            struct second;
            template <template <class...> class L, class T, class U, class... Ts>
            struct second<L<T, U, Ts...>> {
                using type = U;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(second, (class List), (List));
    } // namespace meta
} // namespace gridtools
