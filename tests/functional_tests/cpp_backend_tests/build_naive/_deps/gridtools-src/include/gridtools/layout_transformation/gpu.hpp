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

#include <utility>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/hypercube_iterator.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple_util.hpp"

namespace gridtools {
    namespace impl {
        // compile-time block size due to HIP-Clang bug https://github.com/ROCm-Developer-Tools/HIP/issues/1283
        using block_size_1d_t = integral_constant<uint_t, 8>;

        template <class HyperCube, class DstStrides, class SrcStrides, size_t = tuple_util::size<DstStrides>::value>
        struct outer_loop {
            HyperCube m_hyper_cube;
            DstStrides m_dst_strides;
            SrcStrides m_src_strides;

            template <class T>
            GT_FUNCTION_DEVICE void operator()(T *dst, T const *__restrict__ src) const {
                // TODO this range-based loop does not work on daint in release mode
                // for (auto &&outer : hyper_cube) {
                auto &&e = m_hyper_cube.end();
                for (auto &&i = m_hyper_cube.begin(); i != e; ++i) {
                    T *d = dst;
                    T const *__restrict__ s = src;
                    tuple_util::device::for_each(
                        [&](auto i, auto d_stride, auto s_stride) {
                            d += i * d_stride;
                            s += i * s_stride;
                        },
                        *i,
                        m_dst_strides,
                        m_src_strides);
                    *d = *s;
                }
            }
        };

        template <class HyperCube, class DstStrides, class SrcStrides>
        struct outer_loop<HyperCube, DstStrides, SrcStrides, 0> {
            template <class... Ts>
            outer_loop(Ts &&...) {}

            template <class T>
            GT_FUNCTION_DEVICE void operator()(T *dst, T const *__restrict__ src) const {
                *dst = *src;
            }
        };

        template <class HyperCube, class DstStrides, class SrcStrides>
        outer_loop<HyperCube, DstStrides, SrcStrides> make_outer_loop(
            HyperCube hyper_cube, DstStrides dst_strides, SrcStrides src_strides) {
            return {std::move(hyper_cube), std::move(dst_strides), std::move(src_strides)};
        }

        template <class T, class Dims, class DstStrides, class SrcSrtides, class OuterLoop>
        __global__ void transform_cuda_loop_kernel(T *dst,
            T const *__restrict__ src,
            Dims dims,
            DstStrides dst_strides,
            SrcSrtides src_strides,
            OuterLoop outer_loop) {

            uint_t i = blockIdx.x * block_size_1d_t::value + threadIdx.x;
            if (i >= tuple_util::device::get<0>(dims))
                return;
            uint_t j = blockIdx.y * block_size_1d_t::value + threadIdx.y;
            if (j >= tuple_util::device::get<1>(dims))
                return;
            uint_t k = blockIdx.z * block_size_1d_t::value + threadIdx.z;
            if (k >= tuple_util::device::get<2>(dims))
                return;

            outer_loop(dst + i * tuple_util::device::get<0>(dst_strides) + j * tuple_util::device::get<1>(dst_strides) +
                           k * tuple_util::device::get<2>(dst_strides),
                src + i * tuple_util::device::get<0>(src_strides) + j * tuple_util::device::get<1>(src_strides) +
                    k * tuple_util::device::get<2>(src_strides));
        }

        template <class T, class Dims, class DstStrides, class SrcSrides>
        void transform_gpu_loop(T *dst, T const *src, Dims dims, DstStrides dst_strides, SrcSrides src_strides) {
            dim3 grid_size((tuple_util::get<0>(dims) + block_size_1d_t::value - 1) / block_size_1d_t::value,
                (tuple_util::get<1>(dims) + block_size_1d_t::value - 1) / block_size_1d_t::value,
                (tuple_util::get<2>(dims) + block_size_1d_t::value - 1) / block_size_1d_t::value);
            dim3 block_size(block_size_1d_t::value, block_size_1d_t::value, block_size_1d_t::value);
            transform_cuda_loop_kernel<<<grid_size, block_size>>>(dst,
                src,
                dims,
                dst_strides,
                src_strides,
                make_outer_loop(make_hypercube_view(tuple_util::drop_front<3>(dims)),
                    tuple_util::drop_front<3>(dst_strides),
                    tuple_util::drop_front<3>(src_strides)));
#ifndef NDEBUG
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#else
            GT_CUDA_CHECK(cudaGetLastError());
#endif
        }
    } // namespace impl
} // namespace gridtools
