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

#include <type_traits>

#include "length.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        template <class T>
        using is_empty = std::bool_constant<length<T>::value == 0>;
    } // namespace meta
} // namespace gridtools
