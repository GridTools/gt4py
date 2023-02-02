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

#include <boost/preprocessor/punctuation/remove_parens.hpp>

#define GT_META_DELEGATE_TO_LAZY(fun, signature, args) \
    template <BOOST_PP_REMOVE_PARENS(signature)>       \
    using fun = typename lazy::fun<BOOST_PP_REMOVE_PARENS(args)>::type

/**
 *  NVCC bug workaround: sizeof... works incorrectly within template alias context.
 */
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ < 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6))

namespace gridtools {
    namespace meta {
        template <class... Ts>
        struct sizeof_3_dots : std::integral_constant<std::size_t, sizeof...(Ts)> {};
    } // namespace meta
} // namespace gridtools

#define GT_SIZEOF_3_DOTS(Ts) ::gridtools::meta::sizeof_3_dots<Ts...>::value
#else
#define GT_SIZEOF_3_DOTS(Ts) sizeof...(Ts)
#endif
