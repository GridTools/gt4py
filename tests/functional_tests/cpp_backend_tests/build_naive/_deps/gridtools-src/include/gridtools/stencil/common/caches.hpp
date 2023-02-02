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
        namespace cache_io_policy {
            /**< Read values form the cached field but do not write back */
            struct fill {};
            /**< Write values back the the cached field but do not read in */
            struct flush {};
        } // namespace cache_io_policy

        namespace cache_type {
            // ij caches require synchronization capabilities, as different (i,j) grid points are
            // processed by parallel cores. GPU backend keeps them in shared memory
            struct ij {};
            // processing of all the k elements is done by same thread, so resources for k caches can be private
            // and do not require synchronization. GPU backend uses registers.
            struct k {};
        } // namespace cache_type
    }     // namespace stencil
} // namespace gridtools
