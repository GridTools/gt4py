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
         *   The default list constructor.
         *
         *   Used within the library when it needed to produce something, that satisfy list concept.
         */
        template <class...>
        struct list {
            using type = list;
        };
    } // namespace meta
} // namespace gridtools
