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

#include <cassert>
#include <utility>

#include "common/array.hpp"
#include "common/defs.hpp"
#include "common/tuple_util.hpp"
#include "layout_transformation/cpu.hpp"

#ifdef GT_CUDACC
#include "common/cuda_is_ptr.hpp"
#include "layout_transformation/gpu.hpp"
#endif

namespace gridtools {
    namespace layout_transformation_impl_ {
        template <class Val, size_t... Is>
        array<Val, sizeof...(Is)> extra_elems(Val val, std::index_sequence<Is...>) {
            return {((void)Is, val)...};
        }

        template <class Tup, class Val, std::enable_if_t<tuple_util::size<Tup>::value >= 3, int> = 0>
        Tup extend(Tup tup, Val) {
            return tup;
        }

        template <class Tup,
            size_t N = tuple_util::size<Tup>::value,
            std::enable_if_t<N<3, int> = 0> auto extend(Tup tup, tuple_util::element<N - 1, Tup> val) {
            return tuple_util::concat(std::move(tup), extra_elems(val, std::make_index_sequence<3 - N>()));
        }

#ifdef GT_CUDACC
        template <class T, class Dims, class DstStrides, class SrcSrides>
        void transform_impl(T *dst, T const *src, Dims dims, DstStrides dst_strides, SrcSrides src_strides) {
            assert(is_gpu_ptr(dst) == is_gpu_ptr(src));
            if (is_gpu_ptr(dst))
                impl::transform_gpu_loop(dst, src, std::move(dims), std::move(dst_strides), std::move(src_strides));
            else
                impl::transform_cpu_loop(dst, src, std::move(dims), std::move(dst_strides), std::move(src_strides));
        }
#else
        template <class T, class Dims, class DstStrides, class SrcStrides>
        void transform_impl(T *dst, T const *src, Dims dims, DstStrides dst_strides, SrcStrides src_strides) {
            impl::transform_cpu_loop(dst, src, dims, dst_strides, src_strides);
        }
#endif

        template <class T, class Dims, class DstStrides, class SrcStrides>
        void transform_layout(T *dst, T const *src, Dims dims, DstStrides dst_strides, SrcStrides src_strides) {
            assert(dst);
            assert(src);
            static_assert(tuple_util::size<Dims>::value > 0, "wrong size of Dims");
            static_assert(
                tuple_util::size<Dims>::value == tuple_util::size<DstStrides>::value, "wrong size of DstStrides");
            static_assert(
                tuple_util::size<Dims>::value == tuple_util::size<SrcStrides>::value, "wrong size of SrcStrides");
            transform_impl(dst, src, extend(dims, 1), extend(dst_strides, 0), extend(src_strides, 0));
        }
    } // namespace layout_transformation_impl_
    using layout_transformation_impl_::transform_layout;
} // namespace gridtools
