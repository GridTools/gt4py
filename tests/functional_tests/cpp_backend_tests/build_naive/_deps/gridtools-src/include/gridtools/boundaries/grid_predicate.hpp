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

#include "direction.hpp"

namespace gridtools {
    namespace boundaries {
        /** @brief predicate returning whether I am or not at the global boundary, based on a processor grid
         */
        template <class T>
        struct proc_grid_predicate {
            T const &m_grid;

            proc_grid_predicate(T const &g) : m_grid{g} {}

            template <sign I, sign J, sign K>
            bool operator()(direction<I, J, K>) const {
                return m_grid.proc(I, J, K) == -1;
            }
        };
        template <class T>
        proc_grid_predicate<T> make_proc_grid_predicate(T const &obj) {
            return {obj};
        }
    } // namespace boundaries
} // namespace gridtools
