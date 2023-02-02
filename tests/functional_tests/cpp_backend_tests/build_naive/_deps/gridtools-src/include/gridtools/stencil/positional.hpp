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

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"

namespace gridtools {
    namespace stencil {
        struct positional_stride {};

        // Represents position in the computation space.
        // Models SID concept
        template <class Dim>
        struct positional {
            int_t m_val;

            GT_FUNCTION constexpr positional(int_t val = 0) : m_val{val} {}

            GT_FUNCTION constexpr int operator*() const { return m_val; }
            GT_FUNCTION constexpr positional const &operator()() const { return *this; }
        };

        template <class Dim>
        GT_FUNCTION positional<Dim> operator+(positional<Dim> lhs, positional<Dim> rhs) {
            return {lhs.m_val + rhs.m_val};
        }

        template <class Dim>
        typename hymap::keys<Dim>::template values<positional_stride> sid_get_strides(positional<Dim>) {
            return {};
        }

        template <class Dim>
        GT_FUNCTION void sid_shift(positional<Dim> &p, positional_stride, int_t offset) {
            p.m_val += offset;
        }

        template <class Dim>
        positional<Dim> sid_get_ptr_diff(positional<Dim>);

        template <class Dim>
        positional<Dim> sid_get_origin(positional<Dim> obj) {
            return obj;
        }
    } // namespace stencil
} // namespace gridtools
