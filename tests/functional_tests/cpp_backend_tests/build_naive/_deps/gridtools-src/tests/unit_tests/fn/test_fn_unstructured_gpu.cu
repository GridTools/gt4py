/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/array.hpp>
#include <gridtools/fn/unstructured.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/backend/gpu.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        template <int I>
        using int_t = integral_constant<int, I>;

        template <class C, int MaxNeighbors>
        struct stencil {
            GT_FUNCTION constexpr auto operator()() const {
                return [](auto const &in) {
                    int tmp = 0;
                    tuple_util::host_device::for_each(
                        [&](auto i) {
                            auto shifted = shift(in, C(), i);
                            if (can_deref(shifted))
                                tmp += deref(shifted);
                        },
                        meta::rename<tuple, meta::make_indices_c<MaxNeighbors>>());
                    return tmp;
                };
            }
        };

        struct v2v {};
        struct v2e {};

        using block_sizes_t = meta::list<meta::list<unstructured::dim::horizontal, int_t<32>>,
            meta::list<unstructured::dim::vertical, int_t<1>>>;

        TEST(unstructured, v2v_sum) {
            auto apply_stencil = [](auto executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil<v2v, 3>(), 1_c).execute();
            };
            auto fencil = [&](auto const &v2v_table, int nvertices, int nlevels, auto &out, auto const &in) {
                auto v2v_conn = connectivity<v2v>(v2v_table);
                auto domain = unstructured_domain({nvertices, nlevels}, {}, v2v_conn);
                auto backend = make_backend(backend::gpu<block_sizes_t>(), domain);
                apply_stencil(backend.stencil_executor(), out, in);
            };

            auto v2v_table = cuda_util::cuda_malloc<array<int, 3>>(3);
            int v2v_tableh[3][3] = {{1, 2, -1}, {0, 2, -1}, {0, 1, -1}};
            cudaMemcpy(v2v_table.get(), v2v_tableh, 3 * sizeof(array<int, 3>), cudaMemcpyHostToDevice);

            auto in = cuda_util::cuda_malloc<int>(3 * 5);
            auto out = cuda_util::cuda_malloc<int>(3 * 5);
            int inh[3][5], outh[3][5] = {};
            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k)
                    inh[v][k] = 5 * v + k;
            cudaMemcpy(in.get(), inh, 3 * 5 * sizeof(int), cudaMemcpyHostToDevice);

            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(x))
                    .set<property::strides>(
                        hymap::keys<unstructured::dim::horizontal, unstructured::dim::vertical>::make_values(5_c, 1_c));
            };
            auto in_s = as_synthetic(in.get());
            auto out_s = as_synthetic(out.get());

            GT_CUDA_CHECK(cudaDeviceSynchronize());
            fencil(v2v_table.get(), 3, 5, out_s, in_s);
            GT_CUDA_CHECK(cudaDeviceSynchronize());
            cudaMemcpy(outh, out.get(), 3 * 5 * sizeof(int), cudaMemcpyDeviceToHost);

            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k) {
                    int nbsum = 0;
                    for (int i = 0; i < 3; ++i) {
                        int nb = v2v_tableh[v][i];
                        if (nb != -1)
                            nbsum += inh[nb][k];
                    }
                    EXPECT_EQ(outh[v][k], nbsum);
                }
        }

        TEST(unstructured, v2e_sum) {
            auto apply_stencil = [](auto executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil<v2e, 2>(), 1_c).execute();
            };
            auto fencil = [&](auto const &v2e_table, int nvertices, int nlevels, auto &out, auto const &in) {
                auto v2e_conn = connectivity<v2e>(v2e_table);
                auto domain = unstructured_domain({nvertices, nlevels}, {}, v2e_conn);
                auto backend = make_backend(backend::gpu<block_sizes_t>(), domain);
                apply_stencil(backend.stencil_executor(), out, in);
            };

            auto v2e_table = cuda_util::cuda_malloc<array<int, 2>>(3);
            int v2e_tableh[3][2] = {{0, 2}, {0, 1}, {1, 2}};
            cudaMemcpy(v2e_table.get(), v2e_tableh, 3 * sizeof(array<int, 2>), cudaMemcpyHostToDevice);

            auto in = cuda_util::cuda_malloc<int>(3 * 5);
            auto out = cuda_util::cuda_malloc<int>(3 * 5);
            int inh[3][5], outh[3][5] = {};
            for (int e = 0; e < 3; ++e)
                for (int k = 0; k < 5; ++k)
                    inh[e][k] = 5 * e + k;
            cudaMemcpy(in.get(), inh, 3 * 5 * sizeof(int), cudaMemcpyHostToDevice);

            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(x))
                    .set<property::strides>(
                        hymap::keys<unstructured::dim::horizontal, unstructured::dim::vertical>::make_values(5_c, 1_c));
            };
            auto in_s = as_synthetic(in.get());
            auto out_s = as_synthetic(out.get());

            GT_CUDA_CHECK(cudaDeviceSynchronize());
            fencil(v2e_table.get(), 3, 5, out_s, in_s);
            GT_CUDA_CHECK(cudaDeviceSynchronize());
            cudaMemcpy(outh, out.get(), 3 * 5 * sizeof(int), cudaMemcpyDeviceToHost);

            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k) {
                    int nbsum = 0;
                    for (int i = 0; i < 2; ++i) {
                        int nb = v2e_tableh[v][i];
                        nbsum += inh[nb][k];
                    }
                    EXPECT_EQ(outh[v][k], nbsum);
                }
        }

    } // namespace
} // namespace gridtools::fn
