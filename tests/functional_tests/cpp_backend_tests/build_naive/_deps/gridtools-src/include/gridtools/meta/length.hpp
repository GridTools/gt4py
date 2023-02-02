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

#include <type_traits>

namespace gridtools {
    namespace meta {
        template <class>
        struct length;
        template <template <class...> class L, class... Ts>
        struct length<L<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};
    } // namespace meta
} // namespace gridtools
