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

#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Convert an integer sequence to a list of corresponding integral constants.
         */
        namespace lazy {
            template <class, template <class...> class = list, template <class T, T> class = std::integral_constant>
            struct iseq_to_list;
            template <template <class T, T...> class ISec,
                class Int,
                Int... Is,
                template <class...>
                class L,
                template <class T, T>
                class C>
            struct iseq_to_list<ISec<Int, Is...>, L, C> {
                using type = L<C<Int, Is>...>;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(iseq_to_list,
            (class ISec, template <class...> class L = list, template <class T, T> class C = std::integral_constant),
            (ISec, L, C));
    } // namespace meta
} // namespace gridtools
