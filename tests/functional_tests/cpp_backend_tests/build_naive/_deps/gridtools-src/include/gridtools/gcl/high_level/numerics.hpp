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
    namespace gcl {
        // computes 3^i
        constexpr int static_pow3(int i) { return i ? 3 * static_pow3(i - 1) : 1; }
    } // namespace gcl
} // namespace gridtools
