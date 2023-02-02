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

#include "id.hpp"
#include "internal/inherit.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Find the record in the map.
         *  "mp_" prefix stands for map.
         *
         *  Map is a list of lists, where the first elements of each inner lists (aka keys) are unique.
         *
         *  @return the inner list with a given Key or `void` if not found
         */
        namespace lazy {
            template <class Map, class Key, class Default = void>
            struct mp_find;
            template <class Key, template <class...> class L, class... Ts, class Default>
            struct mp_find<L<Ts...>, Key, Default> {
                template <template <class...> class Elem, class... Vals>
                static Elem<Key, Vals...> select(id<Elem<Key, Vals...>> *);
                static Default select(void *);

                using type = decltype(select((internal::inherit<id<Ts>...> *)0));
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(mp_find, (class Map, class Key, class Default = void), (Map, Key, Default));
    } // namespace meta
} // namespace gridtools
