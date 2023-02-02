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

#include "id.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <class T>
            struct always {
                template <class...>
                struct apply : id<T> {};
            };
        } // namespace lazy
        template <class T>
        struct always {
            template <class...>
            using apply = T;
        };
    } // namespace meta
} // namespace gridtools
