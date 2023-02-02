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

#include <boost/preprocessor/punctuation/remove_parens.hpp>

/**
 *  This macro expands to the code snippet that generates a compiler error that refers to the type(s) `x`
 *
 *  Works also with parameter packs. I.e you can both `GT_META_PRINT_TYPE(SomeType)` and
 * `GT_META_PRINT_TYPE(SomeTypes...)`
 */
#define GT_META_PRINT_TYPE(x) static_assert(::gridtools::meta::debug::type<BOOST_PP_REMOVE_PARENS(x)>::_)

/**
 *  This macro expands to the code snippet that generates a compiler error that refers to the compile time value(s) of
 * the integral type  `x`
 *
 *  Works also with parameter packs. I.e you can both `GT_META_PRINT_VALUE(SomeValue)` and
 * `GT_META_PRINT_VALUE(SomeValues...)`
 */
#define GT_META_PRINT_VALUE(x)                                                                                \
    static_assert(                                                                                            \
        ::gridtools::meta::debug::value<decltype(::gridtools::meta::debug::first(BOOST_PP_REMOVE_PARENS(x))), \
            BOOST_PP_REMOVE_PARENS(x)>::_)

namespace gridtools {
    namespace meta {
        namespace debug {
            template <class T>
            T first(T, ...);

            template <class...>
            struct type {};
            template <class T, T...>
            struct value {};
        } // namespace debug
    }     // namespace meta
} // namespace gridtools
