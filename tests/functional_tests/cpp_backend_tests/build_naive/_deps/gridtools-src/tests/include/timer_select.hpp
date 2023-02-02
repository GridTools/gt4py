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
    class timer_cuda;
    class timer_omp;
    struct timer_dummy;
} // namespace gridtools

// default timer implementation
#if defined(GT_TIMER_CUDA)
#include <gridtools/common/timer/timer_cuda.hpp>
namespace {
    using timer_impl_t = gridtools::timer_cuda;
}
#elif defined(GT_TIMER_OMP)
#include <gridtools/common/timer/timer_omp.hpp>
namespace {
    using timer_impl_t = gridtools::timer_omp;
}
#elif defined(GT_TIMER_DUMMY)
#include <gridtools/common/timer/timer_dummy.hpp>
namespace {
    using timer_impl_t = gridtools::timer_dummy;
}
#endif
