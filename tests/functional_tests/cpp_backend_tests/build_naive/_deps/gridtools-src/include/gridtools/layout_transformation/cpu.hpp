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

#include "../common/hypercube_iterator.hpp"
#include "../common/tuple_util.hpp"

namespace gridtools {
    namespace impl {
        template <class T, class Dims, class DstStrides, class SrcSrides>
        void transform_cpu_loop(
            T *dst, T const *__restrict__ src, Dims dims, DstStrides dst_strides, SrcSrides src_strides) {

            auto omp_loop = [size_i = tuple_util::get<0>(dims),
                                size_j = tuple_util::get<1>(dims),
                                size_k = tuple_util::get<2>(dims),
                                src_stride_i = tuple_util::get<0>(src_strides),
                                src_stride_j = tuple_util::get<1>(src_strides),
                                src_stride_k = tuple_util::get<2>(src_strides),
                                dst_stride_i = tuple_util::get<0>(dst_strides),
                                dst_stride_j = tuple_util::get<1>(dst_strides),
                                dst_stride_k = tuple_util::get<2>(dst_strides)](T *dst, T const *__restrict__ src) {
#pragma omp parallel for collapse(3)
                for (int i = 0; i < size_i; ++i)
                    for (int j = 0; j < size_j; ++j)
                        for (int k = 0; k < size_k; ++k)
                            dst[dst_stride_i * i + dst_stride_j * j + dst_stride_k * k] =
                                src[src_stride_i * i + src_stride_j * j + src_stride_k * k];
            };

            auto offset = [](auto const &index, auto const &strides) {
                size_t res = 0;
                tuple_util::for_each([&res](auto i, auto stride) { res += i * stride; }, index, strides);
                return res;
            };

            auto &&extra_src_strides = tuple_util::drop_front<3>(std::move(src_strides));
            auto &&extra_dst_strides = tuple_util::drop_front<3>(std::move(dst_strides));

            for (auto i : make_hypercube_view(tuple_util::drop_front<3>(dims)))
                omp_loop(dst + offset(i, extra_dst_strides), src + offset(i, extra_src_strides));
        }
    } // namespace impl
} // namespace gridtools
