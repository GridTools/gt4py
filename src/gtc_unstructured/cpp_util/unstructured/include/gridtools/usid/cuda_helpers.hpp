#pragma once

#ifndef __CUDACC__
#error Tried to compile CUDA code with a regular C++ compiler.
#endif

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/pair.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/storage/gpu.hpp>

#include "dim.hpp"
#include "helpers.hpp"

namespace gridtools::usid::cuda {
using traits_t = storage::gpu;

inline auto make_allocator() {
  return sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char[]>);
}

template <class Kernel, class Ptr, class Strides, class... Neighbors>
__global__ void kernel(int h_size, Ptr ptr_holder, Strides strides,
                       Neighbors... neighbors) {
  auto h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= h_size)
    return;
  Kernel()()(
      sid::shifted(ptr_holder(), device::at_key<dim::h>(strides), h), strides,
      tuple_util::device::make<pair>(neighbors.first(), neighbors.second)...);
}

template <class Kernel, class Size, class Sid, class... Sids>
void call_kernel(Size size, Sid &&fields, Sids &&...neighbor_fields) {
  int threads_per_block = 32;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  kernel<Kernel><<<blocks, threads_per_block>>>(
      size, sid::get_origin(fields), sid::get_strides(fields),
      tuple_util::make<pair>(
          sid::get_origin(neighbor_fields),
          at_key<dim::h>(sid::get_strides(neighbor_fields)))...);
  GT_CUDA_CHECK(cudaGetLastError());
}

template <class Tag, class Ptr>
__device__ decltype(auto) field(Ptr const &ptr) {
  return *device::at_key<Tag>(ptr);
}
} // namespace gridtools::usid::cuda
