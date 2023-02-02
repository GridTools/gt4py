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

#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/runtime.hpp>

namespace gridtools {
    namespace thread_pool {
        struct hpx {

            friend auto thread_pool_get_thread_num(hpx) { return ::hpx::get_worker_thread_num(); }
            friend auto thread_pool_get_max_threads(hpx) { return ::hpx::get_num_worker_threads(); }

            template <class F, class I>
            friend void thread_pool_parallel_for_loop(hpx, F const &f, I lim) {
                ::hpx::parallel::for_loop(::hpx::parallel::execution::par, 0, lim, f);
            }
        };
    } // namespace thread_pool
} // namespace gridtools
