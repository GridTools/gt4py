/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/layout_transformation.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/for_each.hpp>
#include <gridtools/common/hypercube_iterator.hpp>
#include <gridtools/meta.hpp>

#ifdef GT_CUDACC
#include <gridtools/common/cuda_util.hpp>
#endif

namespace {
    using namespace gridtools;

    struct host {};

    template <class T, class U, size_t N, size_t M, class Dims, class DstStrides, class SrcSrides>
    void testee(
        host, T (&dst)[N], U (&src)[M], Dims const &dims, DstStrides const &dst_strides, SrcSrides const &src_strides) {
        static_assert(std::is_same_v<std::remove_all_extents_t<T>, std::remove_all_extents_t<U>>);
        using data_t = std::remove_all_extents_t<T>;
        transform_layout((data_t *)dst, (data_t const *)src, dims, dst_strides, src_strides);
    }

#ifdef GT_CUDACC
    struct device {};

    template <class T, class U, size_t N, size_t M, class Dims, class DstStrides, class SrcSrides>
    void testee(device,
        T (&dst)[N],
        U (&src)[M],
        Dims const &dims,
        DstStrides const &dst_strides,
        SrcSrides const &src_strides) {
        static_assert(std::is_same_v<std::remove_all_extents_t<T>, std::remove_all_extents_t<U>>);
        using data_t = std::remove_all_extents_t<T>;
        auto &src_arr = reinterpret_cast<array<U, M> &>(src);
        auto &dst_arr = reinterpret_cast<array<T, N> &>(dst);
        auto d_src = cuda_util::make_clone(src_arr);
        auto d_dst = cuda_util::make_clone(dst_arr);
        transform_layout((data_t *)d_dst.get(), (data_t const *)d_src.get(), dims, dst_strides, src_strides);
        dst_arr = cuda_util::from_clone(d_dst);
    }

    using envs_t = meta::list<host, device>;
#else
    using envs_t = meta::list<host>;
#endif

    TEST(layout_transformation, 3D_reverse_layout) {
        for_each<envs_t>([](auto env) {
            constexpr size_t Nx = 4, Ny = 5, Nz = 6;
            double src[Nx][Ny][Nz];
            double dst[Nz][Ny][Nx];
            auto dims = array{Nx, Ny, Nz};
            for (auto i : make_hypercube_view(dims)) {
                src[i[0]][i[1]][i[2]] = 100 * i[0] + 10 * i[1] + i[2];
                dst[i[2]][i[1]][i[0]] = -1;
            }
            testee(env, dst, src, dims, array{1, Nx, Nx * Ny}, array{Ny * Nz, Nz, 1});
            for (auto i : make_hypercube_view(dims))
                EXPECT_DOUBLE_EQ(dst[i[2]][i[1]][i[0]], src[i[0]][i[1]][i[2]]);
        });
    }

    TEST(layout_transformation, 4D_reverse_layout) {
        for_each<envs_t>([](auto env) {
            constexpr size_t Nx = 4, Ny = 5, Nz = 6, Nw = 7;
            double src[Nx][Ny][Nz][Nw];
            double dst[Nw][Nz][Ny][Nx];
            auto dims = array{Nx, Ny, Nz, Nw};
            for (auto i : make_hypercube_view(dims)) {
                src[i[0]][i[1]][i[2]][i[3]] = 1000 * i[0] + 100 * i[1] + 10 * i[2] + i[3];
                dst[i[3]][i[2]][i[1]][i[0]] = -1;
            }
            testee(env, dst, src, dims, array{1, Nx, Nx * Ny, Nx * Ny * Nz}, array{Ny * Nz * Nw, Nz * Nw, Nw, 1});
            for (auto i : make_hypercube_view(dims))
                EXPECT_DOUBLE_EQ(dst[i[3]][i[2]][i[1]][i[0]], src[i[0]][i[1]][i[2]][i[3]]);
        });
    }

    TEST(layout_transformation, 2D_reverse_layout) {
        for_each<envs_t>([](auto env) {
            constexpr size_t Nx = 4, Ny = 5;
            double src[Nx][Ny];
            double dst[Ny][Nx];
            auto dims = array{Nx, Ny};
            for (auto i : make_hypercube_view(dims)) {
                src[i[0]][i[1]] = 100 * i[0] + 10 * i[1];
                dst[i[1]][i[0]] = -1;
            }
            testee(env, dst, src, dims, array{1, Nx}, array{Ny, 1});
            for (auto i : make_hypercube_view(dims))
                EXPECT_DOUBLE_EQ(dst[i[1]][i[0]], src[i[0]][i[1]]);
        });
    }

    TEST(layout_transformation, 1D_layout_with_stride2) {
        for_each<envs_t>([](auto env) {
            constexpr size_t Nx = 4;
            double src[Nx];
            double dst[Nx][2];
            for (size_t i = 0; i != Nx; ++i) {
                src[i] = i;
                dst[i][0] = dst[i][1] = -1;
            }
            testee(env, dst, src, array{Nx}, array{2}, array{1});
            for (size_t i = 0; i != Nx; ++i) {
                EXPECT_DOUBLE_EQ(dst[i][0], src[i]); // the indexable elements match
                EXPECT_DOUBLE_EQ(dst[i][1], -1);     // the non-indexable are not touched
            }
        });
    }
} // namespace
