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

#include <array>
#include <type_traits>

namespace gridtools {
    namespace gcl {
        template <size_t I>
        class boollist {
            std::array<bool, I> m_value;

          public:
            bool value(uint_t id) const { return m_value[id]; }

            boollist(bool v0) : m_value{v0} {}
            boollist(bool v0, bool v1) : m_value{v0, v1} {}
            boollist(bool v0, bool v1, bool v2) : m_value{v0, v1, v2} {}

            template <typename LayoutMap>
            boollist<LayoutMap::masked_length> permute(std::enable_if_t<LayoutMap::masked_length == 1> * = 0) const {
                return boollist<LayoutMap::masked_length>(m_value[LayoutMap::find(0)]);
            }

            template <typename LayoutMap>
            boollist<LayoutMap::masked_length> permute(std::enable_if_t<LayoutMap::masked_length == 2> * = 0) const {
                return boollist<LayoutMap::masked_length>(m_value[LayoutMap::find(0)], m_value[LayoutMap::find(1)]);
            }

            template <typename LayoutMap>
            boollist<LayoutMap::masked_length> permute(std::enable_if_t<LayoutMap::masked_length == 3> * = 0) const {
                return boollist<LayoutMap::masked_length>(
                    m_value[LayoutMap::find(0)], m_value[LayoutMap::find(1)], m_value[LayoutMap::find(2)]);
            }
        };
    } // namespace gcl
} // namespace gridtools
