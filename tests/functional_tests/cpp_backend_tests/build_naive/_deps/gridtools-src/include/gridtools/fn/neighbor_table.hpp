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

#include "../common/tuple_util.hpp"
#include "../meta/logical.hpp"

/**
 *   Basic API for the neighbor table concept.
 *
 *   The Concept
 *   ===========
 *
 *   A type `T` models the neighbor table concept if it has the following function defined and available via ADL:
 *     `Neighbors neighbor_table_neighbors(T const&, int);`
 *
 *   The return type `Neighbors` should be a tuple-like of integral values.
 *
 *   Pure functional behavior without side-effects is expected from the provided function.
 *
 *   Compile-time API
 *   ================
 *
 *   `neighbor_table::is_neighbor_table<T>`: predicate that checks if T models the neighbor table concept.
 *
 *   Run-time API
 *   ============
 *
 *   Wrapper for the concept function:
 *
 *   `Neighbors neighbor_table::neighbors(NeighborTable const&, int);`
 *
 *   Default Implementation
 *   ======================
 *
 *   For pointers to tuple-likes of integral values, a default implementation exists. That is,
 *   `T const &neighbor_table_neighbors(T const* table, int index);`
 *   is defined for any tuple-like T with all-integral values to return `table[index]`.
 */

namespace gridtools::fn::neighbor_table {
    namespace neighbor_table_impl_ {

        template <class T>
        using is_neighbor_list = std::conjunction<tuple_util::is_tuple_like<T>,
            meta::all_of<std::is_integral, tuple_util::traits::to_types<T>>>;

        template <class T, std::enable_if_t<is_neighbor_list<T>::value, int> = 0>
        GT_FUNCTION T const &neighbor_table_neighbors(T const *table, int index) {
            return table[index];
        }

        template <class NeighborTable>
        GT_FUNCTION constexpr auto neighbors(NeighborTable const &nt, int index)
            -> decltype(neighbor_table_neighbors(nt, index)) {
            return neighbor_table_neighbors(nt, index);
        }

        template <class T>
        using neighbor_list_type = std::remove_cv_t<std::remove_reference_t<
            decltype(::gridtools::fn::neighbor_table::neighbor_table_impl_::neighbors(std::declval<T const &>(), 0))>>;

        template <class T, class = void>
        struct is_neighbor_table : std::false_type {};

        template <class T>
        struct is_neighbor_table<T, std::enable_if_t<is_neighbor_list<neighbor_list_type<T>>::value>> : std::true_type {
        };

    } // namespace neighbor_table_impl_

    using neighbor_table_impl_::is_neighbor_table;
    using neighbor_table_impl_::neighbors;

} // namespace gridtools::fn::neighbor_table
