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

#include "filter.hpp"
#include "first.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        template <class Key>
        struct mp_remove_helper {
            template <class T>
            using apply = std::negation<std::is_same<typename lazy::first<T>::type, Key>>;
        };

        template <class Map, class Key>
        using mp_remove = filter<mp_remove_helper<Key>::template apply, Map>;
    } // namespace meta
} // namespace gridtools
