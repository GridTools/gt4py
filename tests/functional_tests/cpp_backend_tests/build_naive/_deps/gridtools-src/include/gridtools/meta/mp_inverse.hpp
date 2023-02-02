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

#include "clear.hpp"
#include "fold.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "mp_insert.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <class State, class Item>
            struct mp_inverse_helper;

            template <class State, template <class...> class L, class Key, class... Vals>
            struct mp_inverse_helper<State, L<Key, Vals...>>
                : foldl<meta::mp_insert, State, meta::list<L<Vals, Key>...>> {};
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(mp_inverse_helper, (class State, class Item), (State, Item));

        template <class Src>
        using mp_inverse = foldl<mp_inverse_helper, clear<Src>, Src>;
    } // namespace meta
} // namespace gridtools
