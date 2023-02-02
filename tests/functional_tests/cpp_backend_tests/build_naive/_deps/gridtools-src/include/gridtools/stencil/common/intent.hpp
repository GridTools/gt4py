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

#include "../../common/host_device.hpp"

namespace gridtools {
    namespace stencil {
        /**
         * @brief accessor I/O policy
         */
        enum class intent { in, inout };

        template <intent Intent, class T>
        struct apply_intent_type;

        template <class T>
        struct apply_intent_type<intent::inout, T &> {
            using type = T &;
        };

        template <class T>
        struct apply_intent_type<intent::inout, T const &> {};

        template <class T>
        struct apply_intent_type<intent::in, T> {
            using type = T;
        };

        template <class T>
        struct apply_intent_type<intent::in, T &> {
            using type = T const &;
        };

        template <intent Intent, class T>
        using apply_intent_t = typename apply_intent_type<Intent, T>::type;

        template <intent Intent, class T, class Res = typename apply_intent_type<Intent, T>::type>
        GT_FUNCTION Res apply_intent(T &&obj) {
            return static_cast<Res>(obj);
        }
    } // namespace stencil
} // namespace gridtools
