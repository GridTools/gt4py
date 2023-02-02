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

namespace gridtools {
    namespace stencil {
        /**
           This is a syntactic token which is used to declare the public interface of a stencil operator.
           This is used to define the tuple of arguments/accessors that a stencil operator expects.
         */
        template <class...>
        struct make_param_list;
    } // namespace stencil
} // namespace gridtools
