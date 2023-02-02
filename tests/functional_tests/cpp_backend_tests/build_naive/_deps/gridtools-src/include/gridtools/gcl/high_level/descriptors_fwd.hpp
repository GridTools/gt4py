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
    namespace gcl {
        template <typename DataType, typename>
        class hndlr_descriptor_ut;

        template <typename Datatype, typename GridType, typename, typename, typename>
        class hndlr_dynamic_ut;

        template <typename Haloexch, typename proc_layout, typename Gcl_Arch>
        class hndlr_generic;

        template <typename DataType, typename layoutmap, template <typename> class traits>
        struct field_on_the_fly;
    } // namespace gcl
} // namespace gridtools
