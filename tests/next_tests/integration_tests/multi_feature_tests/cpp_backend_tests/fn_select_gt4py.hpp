#pragma once

// COPIED AND SLIGHTLY MODIFIED FROM gritools/test/include/fn_select.hpp
// The gridtools fn_select assumes that dimensions are named
// integral_constant<int, I>, the GT4Py generated code uses structs as dimension
// identifiers.

#include <type_traits>

#include <gridtools/meta.hpp>

// fn backend
#if defined(GT_FN_NAIVE)
#ifndef GT_STENCIL_NAIVE
#define GT_STENCIL_NAIVE
#endif
#ifndef GT_STORAGE_CPU_KFIRST
#define GT_STORAGE_CPU_KFIRST
#endif
#ifndef GT_TIMER_DUMMY
#define GT_TIMER_DUMMY
#endif
#include <gridtools/fn/backend/naive.hpp>
namespace {
template <class> using fn_backend_t = gridtools::fn::backend::naive;
}
#elif defined(GT_FN_GPU)
#ifndef GT_STENCIL_GPU
#define GT_STENCIL_GPU
#endif
#ifndef GT_STORAGE_GPU
#define GT_STORAGE_GPU
#endif
#ifndef GT_TIMER_CUDA
#define GT_TIMER_CUDA
#endif
#include <gridtools/fn/backend/gpu.hpp>
namespace {
template <class block_sizes>
using fn_backend_t = gridtools::fn::backend::gpu<block_sizes>;
} // namespace
#endif

#include "stencil_select.hpp"
#include "storage_select.hpp"
#include "timer_select.hpp"
namespace {
template <class... dims> struct block_sizes_t {
  template <int... sizes>
  using values = gridtools::meta::zip<
      gridtools::meta::list<dims...>,
      gridtools::meta::list<gridtools::integral_constant<int, sizes>...>>;
};
} // namespace
namespace gridtools::fn::backend {
namespace naive_impl_ {
template <class ThreadPool> struct naive_with_threadpool;
template <class ThreadPool>
storage::cpu_kfirst backend_storage_traits(naive_with_threadpool<ThreadPool>);
template <class ThreadPool>
timer_dummy backend_timer_impl(naive_with_threadpool<ThreadPool>);
template <class ThreadPool>
inline char const *backend_name(naive_with_threadpool<ThreadPool> const &) {
  return "naive";
}
} // namespace naive_impl_

namespace gpu_impl_ {
template <class> struct gpu;
template <class BlockSizes>
storage::gpu backend_storage_traits(gpu<BlockSizes>);
template <class BlockSizes> timer_cuda backend_timer_impl(gpu<BlockSizes>);
template <class BlockSizes>
inline char const *backend_name(gpu<BlockSizes> const &) {
  return "gpu";
}
} // namespace gpu_impl_
} // namespace gridtools::fn::backend
