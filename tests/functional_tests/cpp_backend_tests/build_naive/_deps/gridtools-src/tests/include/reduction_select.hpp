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

#include <gridtools/meta.hpp>

// reduction backend
#if defined(GT_REDUCTION_NAIVE)
#ifndef GT_STENCIL_NAIVE
#define GT_STENCIL_NAIVE
#endif
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_DUMMY
#define GT_TIMER_DUMMY
#endif
#include <gridtools/reduction/naive.hpp>
namespace {
    using reduction_backend_t = gridtools::reduction::naive;
}
#elif defined(GT_REDUCTION_CPU)
#ifndef GT_STENCIL_CPU_IFIRST
#define GT_STENCIL_CPU_IFIRST
#endif
#ifndef GT_STORAGE_CPU_IFIRST
#define GT_STORAGE_CPU_IFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/reduction/cpu.hpp>
namespace {
    using reduction_backend_t = gridtools::reduction::cpu;
}
#elif defined(GT_REDUCTION_GPU)
#ifndef GT_STENCIL_GPU
#define GT_STENCIL_GPU
#endif
#ifndef GT_STORAGE_GPU
#define GT_STORAGE_GPU
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/reduction/gpu.hpp>
namespace {
    using reduction_backend_t = gridtools::reduction::gpu;
}
#endif

#include "stencil_select.hpp"
#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    namespace reduction {
        struct naive;
        storage::cpu_kfirst backend_storage_traits(naive);
        timer_dummy backend_timer_impl(naive);
        inline char const *backend_name(naive const &) { return "naive"; }

        struct cpu;
        storage::cpu_ifirst backend_storage_traits(cpu);
        timer_omp backend_timer_impl(cpu);
        inline char const *backend_name(cpu const &) { return "cpu"; }

        namespace gpu_backend {
            struct gpu;
            storage::gpu backend_storage_traits(gpu);
            timer_cuda backend_timer_impl(gpu);
            inline char const *backend_name(gpu const &) { return "gpu"; }
        } // namespace gpu_backend
    }     // namespace reduction
} // namespace gridtools
