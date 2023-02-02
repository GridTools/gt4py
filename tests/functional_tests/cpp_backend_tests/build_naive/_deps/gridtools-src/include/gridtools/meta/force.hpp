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

namespace gridtools {
    namespace meta {
        /**
         *  Remove laziness from a function
         */
        template <template <class...> class F>
        struct force {
            template <class... Args>
            using apply = typename F<Args...>::type;
        };
    } // namespace meta
} // namespace gridtools
