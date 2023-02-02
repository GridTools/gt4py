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

#include "concat.hpp"
#include "rename.hpp"

namespace gridtools {
    /**
     *  Flatten a list of lists.
     *
     *  Note: this function doesn't go recursive. It just concatenates the inner lists.
     */
    namespace meta {
        namespace lazy {
            template <class T>
            using flatten = rename<meta::concat, T>;
        } // namespace lazy
        template <class T>
        using flatten = rename<concat, T>;
    } // namespace meta
} // namespace gridtools
