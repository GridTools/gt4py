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
#include <utility>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"

namespace gridtools {
    namespace stencil {
        namespace global_parameter_impl_ {
            struct ptr_diff {};

            template <class T>
            struct global_parameter {
                static_assert(std::is_trivially_copy_constructible_v<T>, "global parameter should be trivially copyable");

                T m_value;

                constexpr global_parameter(T val) : m_value(std::move(val)) {}

                constexpr GT_FUNCTION global_parameter operator()() const { return *this; }
                constexpr GT_FUNCTION T operator*() const { return m_value; }

                friend constexpr GT_FUNCTION global_parameter operator+(global_parameter obj, ptr_diff) { return obj; }
                friend constexpr global_parameter sid_get_origin(global_parameter const &obj) { return obj; }
                friend constexpr ptr_diff sid_get_ptr_diff(global_parameter) { return {}; }
            };
        } // namespace global_parameter_impl_

        using global_parameter_impl_::global_parameter;

        template <class T>
        [[deprecated("use global_parameter template deduction")]] constexpr global_parameter<T> make_global_parameter(
            T val) {
            return {std::move(val)};
        }
    } // namespace stencil
} // namespace gridtools
