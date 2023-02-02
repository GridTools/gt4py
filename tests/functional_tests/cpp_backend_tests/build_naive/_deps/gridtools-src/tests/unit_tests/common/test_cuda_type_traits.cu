/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/cuda_type_traits.hpp>

#include <gridtools/common/defs.hpp>

namespace gridtools {
    static_assert(is_texture_type<int>::value);
    static_assert(!is_texture_type<bool>::value);
    static_assert(is_texture_type<double>::value);
    static_assert(is_texture_type<uint_t>::value);
} // namespace gridtools
