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

#include <cstddef>
#include <utility>

#include "iseq_to_list.hpp"
#include "length.hpp"
#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            /**
             *  Make a list of integral constants of indices from 0 to N
             */
            template <std::size_t N, template <class...> class L = list>
            using make_indices_c = iseq_to_list<std::make_index_sequence<N>, L>;

            template <class N, template <class...> class L = list>
            using make_indices = iseq_to_list<std::make_index_sequence<N::value>, L>;

            /**
             *  Make a list of integral constants of indices from 0 to length< List >
             */
            template <class List, template <class...> class L = list>
            using make_indices_for = iseq_to_list<std::make_index_sequence<length<List>::value>, L>;
        } // namespace lazy
        template <std::size_t N, template <class...> class L = list>
        using make_indices_c = typename lazy::iseq_to_list<std::make_index_sequence<N>, L>::type;
        template <class N, template <class...> class L = list>
        using make_indices = typename lazy::iseq_to_list<std::make_index_sequence<N::value>, L>::type;
        template <class List, template <class...> class L = list>
        using make_indices_for = typename lazy::iseq_to_list<std::make_index_sequence<length<List>::value>, L>::type;
    } // namespace meta
} // namespace gridtools
