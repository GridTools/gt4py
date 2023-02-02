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
#include "transform.hpp"

namespace gridtools {
    /**
     *  Zip lists
     */
    namespace meta {
        namespace lazy {
            template <class... Lists>
            using zip = transpose<list<Lists...>>;
        }
        template <class... Lists>
        using zip = typename lazy::transpose<list<Lists...>>::type;
    } // namespace meta
} // namespace gridtools
