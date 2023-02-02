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

#include "../../meta/list.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            template <class Plh, class CacheTypes = meta::list<>, class CacheIOPolicies = meta::list<>>
            struct cache_info {
                using plh_t = Plh;
                using cache_types_t = CacheTypes;
                using cache_io_policies_t = CacheIOPolicies;
            };
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
