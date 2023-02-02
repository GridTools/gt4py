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

#include "list.hpp"
#include "macros.hpp"

namespace gridtools::meta {

    /**
     * Maps compile time values to a type.
     *
     * Motivation:
     *  In `meta` library types are first class citizens, values are not.
     *  All algorithms  are defined in terms of types.
     *  If you need a compile time map for example and you want to store values in there you can use `val`:
     *
     *  // maps arity to functions
     *  using my_map_t = list<
     *      list<val<1>, val<[](auto x) { return 42; }>>,
     *      list<val<2>, val<[](auto x, auto y) { return 88; }>>
     *  >;
     *
     *  auto foo(auto... args) {
     *      // dispatch on the arity of args...;
     *      return second<mp_find<my_map_t, val<sizeof...(args)>>>::value(args...);
     *  }
     *
     */
    template <auto...>
    struct val {};

    template <auto... Vs>
    constexpr val<Vs...> constant = {};

    template <auto V>
    struct val<V> {
        using type = decltype(V);
        static constexpr type value = V;
    };

    /**
     * `vl_split` splits `val<V0, V1, V2>` into the type list:  `list<val<V0>, val<V1>, val<V2>>`.
     * `vl_merge` does the opposite.
     *
     * Useful if you need to process values within the `val` in compile time using some `meta` infrastructure.
     * Example:
     *
     * template <class T>
     * using dedup_val = vl_merge<dedup<vl_split<T>>>;
     *
     */
    namespace lazy {
        template <class>
        struct vl_split;

        template <template <auto...> class H, auto... Vs>
        struct vl_split<H<Vs...>> {
            using type = list<H<Vs>...>;
        };

        template <class>
        struct vl_merge;

        template <template <class...> class L, template <auto...> class H, auto... Vs>
        struct vl_merge<L<H<Vs>...>> {
            using type = H<Vs...>;
        };

    } // namespace lazy
    GT_META_DELEGATE_TO_LAZY(vl_split, class T, T);
    GT_META_DELEGATE_TO_LAZY(vl_merge, class T, T);
} // namespace gridtools::meta
