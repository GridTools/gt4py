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

// stencil backend
#if defined(GT_STENCIL_CPU_KFIRST)
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil/cpu_kfirst.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cpu_kfirst<>;
}
#elif defined(GT_STENCIL_CPU_KFIRST_HPX)
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil/cpu_kfirst.hpp>
#include <gridtools/thread_pool/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/apply.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cpu_kfirst<gridtools::integral_constant<int, 8>,
        gridtools::integral_constant<int, 8>,
        gridtools::thread_pool::hpx>;
}
#elif defined(GT_STENCIL_NAIVE)
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_DUMMY
#define GT_TIMER_DUMMY
#endif
#include <gridtools/stencil/naive.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::naive;
}
#elif defined(GT_STENCIL_CPU_IFIRST)
#ifndef GT_STORAGE_CPU_IFIRST
#define GT_STORAGE_CPU_IFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil/cpu_ifirst.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cpu_ifirst<>;
}
#elif defined(GT_STENCIL_CPU_IFIRST_HPX)
#ifndef GT_STORAGE_CPU_IFIRST
#define GT_STORAGE_CPU_IFIRST
#endif
#ifndef GT_TIMER_OMP
#define GT_TIMER_OMP
#endif
#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/thread_pool/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/apply.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::cpu_ifirst<gridtools::thread_pool::hpx>;
}
#elif defined(GT_STENCIL_GPU)
#ifndef GT_STORAGE_GPU
#define GT_STORAGE_GPU
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/stencil/gpu.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::gpu<>;
}
#elif defined(GT_STENCIL_GPU_HORIZONTAL)
#ifndef GT_STORAGE_GPU
#define GT_STORAGE_GPU
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/stencil/gpu_horizontal.hpp>
namespace {
    using stencil_backend_t = gridtools::stencil::gpu_horizontal<>;
}
#endif

#include "storage_select.hpp"
#include "timer_select.hpp"

#if defined(GT_STENCIL_CPU_KFIRST_HPX) || defined(GT_STENCIL_CPU_IFIRST_HPX)
inline bool &hpx_started() {
    static bool res = false;
    return res;
}

inline void hpx_start(int argc, char **argv) {
    if (hpx_started())
        return;
    ::hpx::start(nullptr, argc, argv);
    hpx_started() = true;
}

inline void hpx_stop() {
    if (!hpx_started())
        return;
    ::hpx::apply([] { return hpx::finalize(); });
    ::hpx::stop();
    hpx_started() = false;
}
#endif

namespace gridtools {
    namespace stencil {

        struct naive;
        storage::cpu_kfirst backend_storage_traits(naive);
        timer_dummy backend_timer_impl(naive);
        inline char const *backend_name(naive const &) { return "naive"; }

        namespace cpu_kfirst_backend {
            template <class, class, class>
            struct cpu_kfirst;

            template <class I, class J, class T>
            storage::cpu_kfirst backend_storage_traits(cpu_kfirst<I, J, T>);

            template <class I, class J, class T>
            timer_omp backend_timer_impl(cpu_kfirst<I, J, T>);

            template <class I, class J, class T>
            char const *backend_name(cpu_kfirst<I, J, T> const &) {
                return "cpu_kfirst";
            }

#if defined(GT_STENCIL_CPU_KFIRST_HPX)
            template <class I, class J>
            char const *backend_name(cpu_kfirst<I, J, thread_pool::hpx> const &) {
                return "cpu_kfirst_hpx";
            }

            template <class I, class J>
            void backend_init(cpu_kfirst<I, J, thread_pool::hpx>, int &argc, char **argv) {
                hpx_start(argc, argv);
            }

            template <class I, class J>
            void backend_finalize(cpu_kfirst<I, J, thread_pool::hpx>) {
                hpx_stop();
            }
#endif
        } // namespace cpu_kfirst_backend

        namespace cpu_ifirst_backend {
            template <class>
            struct cpu_ifirst;

            template <class T>
            storage::cpu_ifirst backend_storage_traits(cpu_ifirst<T>);

            template <class T>
            std::false_type backend_supports_icosahedral(cpu_ifirst<T>);

            template <class T>
            timer_omp backend_timer_impl(cpu_ifirst<T>);

            template <class T>
            char const *backend_name(cpu_ifirst<T> const &) {
                return "cpu_ifirst";
            }

#if defined(GT_STENCIL_CPU_IFIRST_HPX)
            inline char const *backend_name(cpu_ifirst<thread_pool::hpx> const &) { return "cpu_ifirst_hpx"; }

            inline void backend_init(cpu_ifirst<thread_pool::hpx>, int &argc, char **argv) { hpx_start(argc, argv); }

            inline void backend_finalize(cpu_ifirst<thread_pool::hpx>) { hpx_stop(); }
#endif
        } // namespace cpu_ifirst_backend

        namespace gpu_backend {
            template <class, class, class>
            struct gpu;

            template <class I, class J, class K>
            storage::gpu backend_storage_traits(gpu<I, J, K>);

            template <class I, class J, class K>
            timer_cuda backend_timer_impl(gpu<I, J, K>);

            template <class I, class J, class K>
            char const *backend_name(gpu<I, J, K> const &) {
                return "gpu";
            }
        } // namespace gpu_backend

        namespace gpu_horizontal_backend {
            template <class, class, class>
            struct gpu_horizontal;

            template <class I, class J, class K>
            storage::gpu backend_storage_traits(gpu_horizontal<I, J, K>);

            template <class I, class J, class K>
            std::false_type backend_supports_vertical_stencils(gpu_horizontal<I, J, K>);

            template <class I, class J, class K>
            timer_cuda backend_timer_impl(gpu_horizontal<I, J, K>);

            template <class I, class J, class K>
            char const *backend_name(gpu_horizontal<I, J, K> const &) {
                return "gpu_horizontal";
            }
        } // namespace gpu_horizontal_backend
    }     // namespace stencil
} // namespace gridtools
