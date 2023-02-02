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

#include <gridtools/gcl/low_level/arch.hpp>

#if defined(GT_GCL_GPU)
#ifndef GT_STORAGE_GPU
#define GT_STORAGE_GPU
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
namespace {
    using gcl_arch_t = gridtools::gcl::gpu;
}
#elif defined(GT_GCL_CPU)
#ifndef GT_STORAGE_CPU_IFIRST
#define GT_STORAGE_CPU_IFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
namespace {
    using gcl_arch_t = gridtools::gcl::cpu;
}
#endif

#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    namespace gcl {
        storage::cpu_ifirst backend_storage_traits(cpu const &);
        timer_omp backend_timer_impl(cpu const &);
        inline char const *backend_name(cpu const &) { return "cpu"; }

        storage::gpu backend_storage_traits(gpu const &);
        timer_cuda backend_timer_impl(gpu const &);
        inline char const *backend_name(gpu const &) { return "gpu"; }
    } // namespace gcl
} // namespace gridtools
