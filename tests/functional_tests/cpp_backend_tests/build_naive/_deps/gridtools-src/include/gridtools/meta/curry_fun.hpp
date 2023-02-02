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
        /**
         *  Partially apply high order function F with provided argument G
         */
        template <template <template <class...> class, class...> class F, template <class...> class G>
        struct curry_fun {
            template <class... Args>
            using apply = F<G, Args...>;
        };
    } // namespace meta
} // namespace gridtools
