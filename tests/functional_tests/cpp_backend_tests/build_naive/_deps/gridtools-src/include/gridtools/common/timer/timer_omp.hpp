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

#include "../omp.hpp"

namespace gridtools {
    /**
     * @class timer_omp
     */
    class timer_omp {
        double m_startTime;

      public:
        void start_impl() { m_startTime = omp_get_wtime(); }
        double pause_impl() { return omp_get_wtime() - m_startTime; }
    };
} // namespace gridtools
