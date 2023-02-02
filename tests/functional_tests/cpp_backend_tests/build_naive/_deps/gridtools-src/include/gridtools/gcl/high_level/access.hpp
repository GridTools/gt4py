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
        inline int access(int i1, int i2, int i3, int N1, int N2) { return i1 + i2 * N1 + i3 * N1 * N2; }
    } // namespace gcl
} // namespace gridtools
