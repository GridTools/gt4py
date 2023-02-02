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

#include <cstddef>

#include "concat.hpp"
#include "drop_front.hpp"
#include "macros.hpp"
#include "push_front.hpp"
#include "take.hpp"

namespace gridtools {
    namespace meta {
        template <size_t N, class List, class... Ts>
        using insert_c = concat<take_c<N, List>, push_front<drop_front_c<N, List>, Ts...>>;

        template <class N, class List, class... Ts>
        using insert = concat<take_c<N::value, List>, push_front<drop_front_c<N::value, List>, Ts...>>;
    } // namespace meta
} // namespace gridtools
