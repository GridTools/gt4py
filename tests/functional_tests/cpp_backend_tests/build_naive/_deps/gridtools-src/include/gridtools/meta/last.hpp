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

#include "at.hpp"
#include "length.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <class List>
            using last = at_c<List, length<List>::value - 1>;
        }
        template <class List>
        using last = typename lazy::at_c<List, length<List>::value - 1>::type;
    } // namespace meta
} // namespace gridtools
