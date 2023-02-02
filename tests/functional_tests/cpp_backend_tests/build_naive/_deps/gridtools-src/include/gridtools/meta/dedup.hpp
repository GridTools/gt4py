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

#include "clear.hpp"
#include "fold.hpp"
#include "if.hpp"
#include "macros.hpp"
#include "push_back.hpp"
#include "st_contains.hpp"

namespace gridtools {
    namespace meta {
        // internals
        template <class S, class T>
        using dedup_step_impl = if_c<st_contains<S, T>::value, S, typename lazy::push_back<S, T>::type>;

        /**
         *  Removes duplicates from the List.
         */
        namespace lazy {
            template <class List>
            using dedup = foldl<dedup_step_impl, typename clear<List>::type, List>;
        }
        template <class List>
        using dedup = typename lazy::foldl<dedup_step_impl, typename lazy::clear<List>::type, List>::type;
    } // namespace meta
} // namespace gridtools
