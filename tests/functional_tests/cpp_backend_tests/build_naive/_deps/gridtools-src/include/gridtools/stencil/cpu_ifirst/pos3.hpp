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
    namespace stencil {
        namespace cpu_ifirst_backend {
            template <class T>
            struct pos3 {
                T i, j, k;
            };

            template <class T>
            constexpr pos3<T> make_pos3(T i, T j, T k) {
                return {i, j, k};
            }
        } // namespace cpu_ifirst_backend
    }     // namespace stencil
} // namespace gridtools
