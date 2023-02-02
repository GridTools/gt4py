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
         *  Partially apply function F with provided arguments BoundArgs
         *
         *  Note:  if `BoundArgs...` is empty this function just converts a function to the meta class. Like mpl::quote
         */
        template <template <class...> class F, class... BoundArgs>
        struct curry {
            template <class... Args>
            using apply = F<BoundArgs..., Args...>;
        };
    } // namespace meta
} // namespace gridtools
