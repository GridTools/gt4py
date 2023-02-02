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

#include "length.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "make_indices.hpp"
#include "mp_find.hpp"
#include "second.hpp"
#include "zip.hpp"

namespace gridtools {
    namespace meta {
        /**
         * return the position of T in the Set. If there is no T, it returns the length of the Set.
         *
         *  @pre All elements in Set are different.
         */
        template <class Set, class T>
        struct st_position
            : lazy::second<mp_find<zip<Set, make_indices_for<Set>>, T, meta::list<void, length<Set>>>>::type {};
    } // namespace meta
} // namespace gridtools
