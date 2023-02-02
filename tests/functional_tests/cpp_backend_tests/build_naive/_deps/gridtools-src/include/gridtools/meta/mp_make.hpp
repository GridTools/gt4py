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
#include "dedup.hpp"
#include "first.hpp"
#include "flatten.hpp"
#include "force.hpp"
#include "list.hpp"
#include "rename.hpp"
#include "transform.hpp"

namespace gridtools {
    namespace meta {
        template <class...>
        struct pick_item_impl;

        template <class Key, class Item>
        struct pick_item_impl<Key, Item> {
            using type = list<>;
        };

        template <class Key, template <class...> class L, class... Vals>
        struct pick_item_impl<Key, L<Key, Vals...>> {
            using type = list<L<Key, Vals...>>;
        };

        template <template <class...> class MergeItems, class Items>
        struct merge_items_impl_f {
            template <class Key>
            using apply =
                rename<MergeItems, flatten<transform<curry<force<pick_item_impl>::apply, Key>::template apply, Items>>>;
        };

        /**
         *  Construct a map from the items.
         *  The keys of the items don't have to be unique.
         *  In the case of non unique keys all items with the same key are merged with the provided `MergeItems`
         *  function.
         */
        template <template <class...> class MergeItems, class Items>
        using mp_make =
            transform<merge_items_impl_f<MergeItems, Items>::template apply, dedup<transform<first, Items>>>;
    } // namespace meta
} // namespace gridtools
