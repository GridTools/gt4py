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

#include "../common/layout_map.hpp"

namespace gridtools {
    namespace boundaries {
        /** \ingroup Distributed-Boundaries
         * @{ */

        template <typename StorageType, typename Arch, typename TimerImpl>
        struct comm_traits {
            using proc_layout = layout_map<0, 1, 2>;
            using comm_arch_type = Arch;
            using timer_impl_t = TimerImpl;
            using data_layout = typename StorageType::element_type::layout_t;
            using value_type = typename StorageType::element_type::data_t;
        };
        /** @} */
    } // namespace boundaries
} // namespace gridtools
