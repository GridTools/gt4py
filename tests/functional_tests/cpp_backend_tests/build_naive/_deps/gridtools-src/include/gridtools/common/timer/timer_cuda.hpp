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

#include "../cuda_runtime.hpp"
#include "../cuda_util.hpp"

namespace gridtools {
    /**
     * @class timer_cuda
     */
    class timer_cuda {
        struct destroy_event {
            using pointer = cudaEvent_t;
            void operator()(cudaEvent_t event) const { cudaEventDestroy(event); }
        };

        using event_holder = std::unique_ptr<void, destroy_event>;

        static event_holder create_event() {
            cudaEvent_t event;
            GT_CUDA_CHECK(cudaEventCreate(&event));
            return event_holder(event);
        }

        event_holder m_start = create_event();
        event_holder m_stop = create_event();

      public:
        void start_impl() {
            // insert a start event
            GT_CUDA_CHECK(cudaEventRecord(m_start.get(), 0));
        }

        double pause_impl() {
            // insert stop event and wait for it
            GT_CUDA_CHECK(cudaEventRecord(m_stop.get(), 0));
            GT_CUDA_CHECK(cudaEventSynchronize(m_stop.get()));

            // compute the timing
            float result;
            GT_CUDA_CHECK(cudaEventElapsedTime(&result, m_start.get(), m_stop.get()));
            return result * 0.001; // convert ms to s
        }
    };
} // namespace gridtools
