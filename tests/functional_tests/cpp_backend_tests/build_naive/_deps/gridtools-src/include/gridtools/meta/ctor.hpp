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

#include "curry.hpp"
#include "defer.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Extracts "producing template" from the list.
         *
         *  I.e ctor<some_instantiation_of_std_tuple>::apply is an alias of std::tuple.
         */
        namespace lazy {
            template <class>
            struct ctor;
            template <template <class...> class L, class... Ts>
            struct ctor<L<Ts...>> : defer<L> {};
        } // namespace lazy
        template <class>
        struct ctor;
        template <template <class...> class L, class... Ts>
        struct ctor<L<Ts...>> : curry<L> {};
    } // namespace meta
} // namespace gridtools
