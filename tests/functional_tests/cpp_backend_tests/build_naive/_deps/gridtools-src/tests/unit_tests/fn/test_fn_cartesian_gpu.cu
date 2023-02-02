/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/cartesian.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/backend/gpu.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        template <int I>
        using int_t = integral_constant<int, I>;

        struct stencil {
            constexpr auto operator()() const {
                using namespace cartesian::dim;
                return [](auto const &in) { return deref(shift(in, i(), 1_c)); };
            }
        };

        struct fwd_sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return acc + deref(iter); }, [](auto acc) { return acc; });
            }
        };

        struct bwd_sum_scan : bwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return acc + deref(iter); }, [](auto acc) { return acc; });
            }
        };

        TEST(cartesian, stencil) {
            using block_sizes_t = meta::list<meta::list<cartesian::dim::i, int_t<32>>,
                meta::list<cartesian::dim::j, int_t<8>>,
                meta::list<cartesian::dim::k, int_t<1>>>;
            auto apply_stencil = [](auto executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil(), 1_c).execute();
            };

            auto fencil = [&](auto const &sizes, auto &out, auto const &in) {
                auto be = backend::gpu<block_sizes_t>();
                auto alloc = tmp_allocator(be);
                auto tmp = allocate_global_tmp<int>(alloc, sizes);
                auto domain = cartesian_domain(std::array<int, 3>{sizes[0] - 1, sizes[1], sizes[2]});
                auto backend = make_backend(be, domain);
                apply_stencil(backend.stencil_executor(), tmp, in);
                apply_stencil(backend.stencil_executor(), out, tmp);
            };

            auto in = cuda_util::cuda_malloc<int>(5 * 3 * 2);
            auto out = cuda_util::cuda_malloc<int>(5 * 3 * 2);
            int inh[5][3][2], outh[5][3][2] = {};
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        inh[i][j][k] = 6 * i + 2 * j + k;
            cudaMemcpy(in.get(), inh, 5 * 3 * 2 * sizeof(int), cudaMemcpyHostToDevice);
            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(x))
                    .set<property::strides>(tuple(6_c, 2_c, 1_c));
            };

            auto out_s = as_synthetic(out.get());
            auto in_s = as_synthetic(in.get());
            fencil(std::array{5, 3, 2}, out_s, in_s);

            cudaMemcpy(outh, out.get(), 5 * 3 * 2 * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        EXPECT_EQ(outh[i][j][k], 6 * (i + 2) + 2 * j + k);
        }

        TEST(cartesian, vertical) {
            using block_sizes_t = meta::list<meta::list<cartesian::dim::i, int_t<32>>,
                meta::list<cartesian::dim::j, int_t<8>>,
                meta::list<cartesian::dim::k, int_t<1>>>;
            auto apply_double_scan = [](auto executor, auto &a, auto &b, auto const &c) {
                executor()
                    .arg(a)
                    .arg(b)
                    .arg(c)
                    .assign(1_c, fwd_sum_scan(), 42, 2_c)
                    .assign(0_c, bwd_sum_scan(), 8, 1_c)
                    .execute();
            };

            auto double_scan = [&](auto sizes, auto &a, auto &b, auto const &c) {
                auto domain = cartesian_domain(sizes);
                auto backend = make_backend(backend::gpu<block_sizes_t>(), domain);
                apply_double_scan(backend.vertical_executor(), a, b, c);
            };

            std::array<int, 3> sizes = {5, 3, 2};
            auto a = cuda_util::cuda_malloc<int>(5 * 3 * 2);
            auto b = cuda_util::cuda_malloc<int>(5 * 3 * 2);
            auto c = cuda_util::cuda_malloc<int>(5 * 3 * 2);
            int ah[5][3][2] = {}, bh[5][3][2] = {}, ch[5][3][2];
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        ch[i][j][k] = 6 * i + 2 * j + k;
            cudaMemcpy(c.get(), ch, 5 * 3 * 2 * sizeof(int), cudaMemcpyHostToDevice);
            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(x))
                    .set<property::strides>(tuple(6_c, 2_c, 1_c));
            };

            auto a_s = as_synthetic(a.get());
            auto b_s = as_synthetic(b.get());
            auto c_s = as_synthetic(c.get());
            double_scan(sizes, a_s, b_s, c_s);
            cudaMemcpy(bh, b.get(), 5 * 3 * 2 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(ah, a.get(), 5 * 3 * 2 * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j) {
                    int res = 42;
                    for (int k = 0; k < 2; ++k) {
                        res += ch[i][j][k];
                        EXPECT_EQ(bh[i][j][k], res);
                    }
                    res = 8;
                    for (int k = 1; k >= 0; --k) {
                        res += bh[i][j][k];
                        EXPECT_EQ(ah[i][j][k], res);
                    }
                }
        }
    } // namespace
} // namespace gridtools::fn
