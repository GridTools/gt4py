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

#include "macros.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            /**
             *  Normalized std::conditional version, which is proper function in the terms of meta library.
             *
             *  Note: `std::conditional` should be named `if_c` according to `meta` name convention.
             */
            template <class Cond, class Lhs, class Rhs>
            using if_ = std::conditional<Cond::value, Lhs, Rhs>;

            template <bool Cond, class Lhs, class Rhs>
            using if_c = std::conditional<Cond, Lhs, Rhs>;
        } // namespace lazy
        template <class Cond, class Lhs, class Rhs>
        using if_ = std::conditional_t<Cond::value, Lhs, Rhs>;

        template <bool Cond, class Lhs, class Rhs>
        using if_c = std::conditional_t<Cond, Lhs, Rhs>;
    } // namespace meta
} // namespace gridtools
