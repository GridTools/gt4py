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
         *  Add laziness to a function
         */
        template <template <class...> class F>
        struct defer {
            template <class... Args>
            struct apply {
                using type = F<Args...>;
            };
        };
    } // namespace meta
} // namespace gridtools
