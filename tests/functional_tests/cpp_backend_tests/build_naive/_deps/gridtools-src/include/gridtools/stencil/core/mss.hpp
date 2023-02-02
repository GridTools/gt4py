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
            template <class ExecutionEngine, class EsfDescrSequence, class CacheMap>
            struct mss_descriptor {
                using execution_engine_t = ExecutionEngine;
                using esf_sequence_t = EsfDescrSequence;
                using cache_map_t = CacheMap;
            };
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
