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

#include "at.hpp"
#include "id.hpp"
#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        template <std::size_t>
        struct placeholder;

        using _1 = placeholder<0>;
        using _2 = placeholder<1>;
        using _3 = placeholder<2>;
        using _4 = placeholder<3>;
        using _5 = placeholder<4>;
        using _6 = placeholder<5>;
        using _7 = placeholder<6>;
        using _8 = placeholder<7>;
        using _9 = placeholder<8>;
        using _10 = placeholder<9>;

        template <class Arg, class... Params>
        struct replace_placeholders_impl : lazy::id<Arg> {};

        template <std::size_t I, class... Params>
        struct replace_placeholders_impl<placeholder<I>, Params...> : lazy::at_c<list<Params...>, I> {};

        /**
         *  bind for functions
         */
        template <template <class...> class F, class... BoundArgs>
        struct bind {
            template <class... Params>
            using apply = F<typename replace_placeholders_impl<BoundArgs, Params...>::type...>;
        };
    } // namespace meta
} // namespace gridtools
