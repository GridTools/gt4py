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
         *  Identity
         */
        namespace lazy {
            template <class T>
            struct id {
                using type = T;
            };
        } // namespace lazy
        template <class T>
        using id = T;
    } // namespace meta
} // namespace gridtools
