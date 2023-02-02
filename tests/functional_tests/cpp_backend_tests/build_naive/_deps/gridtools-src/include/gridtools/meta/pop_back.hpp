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
#include "pop_front.hpp"
#include "reverse.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <class List>
            using pop_back = reverse<typename pop_front<typename reverse<List>::type>::type>;
        }
        template <class List>
        using pop_back =
            typename lazy::reverse<typename lazy::pop_front<typename lazy::reverse<List>::type>::type>::type;
    } // namespace meta
} // namespace gridtools
