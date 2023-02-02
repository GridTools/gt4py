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

#include "cuda_runtime.hpp"

#include "../meta/dedup.hpp"
#include "../meta/list.hpp"
#include "../meta/st_contains.hpp"

namespace gridtools {
    namespace impl_ {
        using texture_types = meta::dedup<meta::list<char,
            short,
            int,
            long long,
            unsigned char,
            unsigned short,
            unsigned int,
            unsigned long long,
            int2,
            int4,
            uint2,
            uint4,
            float,
            float2,
            float4,
            double,
            double2>>;
    } // namespace impl_

    template <class T>
    using is_texture_type = meta::st_contains<impl_::texture_types, T>;

} // namespace gridtools
