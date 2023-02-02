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
        namespace core {
            /**
             * @brief Descriptors for Elementary Stencil Function (ESF)
             */
            template <class EsfFunction, class Args, class Extent>
            struct esf_descriptor {
                using esf_function_t = EsfFunction;
                using args_t = Args;
                using extent_t = Extent;
            };
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
