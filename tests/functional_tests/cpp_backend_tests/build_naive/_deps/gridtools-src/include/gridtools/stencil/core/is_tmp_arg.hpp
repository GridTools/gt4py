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

#include "../../meta.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            template <class T, class = void>
            struct is_tmp_arg : std::false_type {};

            template <class T>
            struct is_tmp_arg<T, std::void_t<typename T::tmp_tag>> : std::true_type {};
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
