/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/backend/gpu.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/column_stage.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools::fn::backend {
    namespace {
        using namespace literals;
        using sid::property;

        template <int I>
        using int_t = integral_constant<int, I>;

        struct sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return tuple(get<0>(acc) + *iter, get<1>(acc) * *iter); },
                    [](auto acc) { return get<0>(acc); });
            }
        };

        struct make_iterator_mock {
            GT_FUNCTION auto operator()() const {
                return
                    [](auto tag, auto const &ptr, auto const &strides) { return device::at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(backend_gpu, apply_column_stage) {
            auto in = cuda_util::cuda_malloc<int>(5 * 7 * 3);
            auto out = cuda_util::cuda_malloc<int>(5 * 7 * 3);
            int inh[5][7][3], outh[5][7][3] = {};
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 7; ++j)
                    for (int k = 0; k < 3; ++k)
                        inh[i][j][k] = 21 * i + 3 * j + k;
            cudaMemcpy(in.get(), inh, 5 * 7 * 3 * sizeof(int), cudaMemcpyHostToDevice);

            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(x))
                    .set<property::strides>(tuple(21_c, 3_c, 1_c));
            };

            auto composite =
                sid::composite::keys<int_t<0>, int_t<1>>::make_values(as_synthetic(out.get()), as_synthetic(in.get()));

            auto sizes = hymap::keys<int_t<0>, int_t<1>, int_t<2>>::values<int_t<5>, int_t<7>, int_t<3>>();

            column_stage<int_t<1>, sum_scan, 0, 1> cs;

            using block_sizes_t = meta::list<meta::list<int_t<0>, int_t<4>>, meta::list<int_t<2>, int_t<2>>>;

            apply_column_stage(
                gpu<block_sizes_t>(), sizes, cs, make_iterator_mock(), composite, int_t<1>(), tuple(42, 1));

            cudaMemcpy(outh, out.get(), 5 * 7 * 3 * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 5; ++i)
                for (int k = 0; k < 3; ++k) {
                    int res = 42;
                    for (int j = 0; j < 7; ++j) {
                        res += inh[i][j][k];
                        EXPECT_EQ(outh[i][j][k], res);
                    }
                }
        }

        TEST(backend_gpu, apply_column_stage_1d) {
            auto in = cuda_util::cuda_malloc<int>(5);
            auto out = cuda_util::cuda_malloc<int>(5);
            int inh[5], outh[5] = {};
            for (int i = 0; i < 5; ++i)
                inh[i] = i;
            cudaMemcpy(in.get(), inh, 5 * sizeof(int), cudaMemcpyHostToDevice);

            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(x))
                    .set<property::strides>(tuple(1_c));
            };

            auto composite =
                sid::composite::keys<int_t<0>, int_t<1>>::make_values(as_synthetic(out.get()), as_synthetic(in.get()));

            auto sizes = hymap::keys<int_t<0>>::values<int_t<5>>();

            column_stage<int_t<0>, sum_scan, 0, 1> cs;

            using block_sizes_t = meta::list<meta::list<int_t<0>, int_t<4>>, meta::list<int_t<2>, int_t<2>>>;

            apply_column_stage(
                gpu<block_sizes_t>(), sizes, cs, make_iterator_mock(), composite, int_t<0>(), tuple(42, 1));

            cudaMemcpy(outh, out.get(), 5 * sizeof(int), cudaMemcpyDeviceToHost);
            int res = 42;
            for (int i = 0; i < 5; ++i) {
                res += inh[i];
                EXPECT_EQ(outh[i], res);
            }
        }

        TEST(backend_gpu, apply_column_stage_5d) {
            auto in = cuda_util::cuda_malloc<int>(5 * 7 * 3 * 2 * 3);
            auto out = cuda_util::cuda_malloc<int>(5 * 7 * 3 * 2 * 3);
            int inh[5][7][3][2][3], outh[5][7][3][2][3] = {};
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 7; ++j)
                    for (int k = 0; k < 3; ++k)
                        for (int l = 0; l < 2; ++l)
                            for (int m = 0; m < 3; ++m)
                                inh[i][j][k][l][m] = 126 * i + 18 * j + 6 * k + 3 * l + m;
            cudaMemcpy(in.get(), inh, 5 * 7 * 3 * 2 * 3 * sizeof(int), cudaMemcpyHostToDevice);

            auto as_synthetic = [](int *x) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(x))
                    .set<property::strides>(tuple(126_c, 18_c, 6_c, 3_c, 1_c));
            };

            auto composite =
                sid::composite::keys<int_t<0>, int_t<1>>::make_values(as_synthetic(out.get()), as_synthetic(in.get()));

            auto sizes = hymap::keys<int_t<0>, int_t<1>, int_t<2>, int_t<3>, int_t<4>>::
                values<int_t<5>, int_t<7>, int_t<3>, int_t<2>, int_t<3>>();

            column_stage<int_t<1>, sum_scan, 0, 1> cs;

            using block_sizes_t = meta::list<meta::list<int_t<0>, int_t<4>>,
                meta::list<int_t<2>, int_t<2>>,
                meta::list<int_t<3>, int_t<2>>,
                meta::list<int_t<4>, int_t<1>>>;

            apply_column_stage(
                gpu<block_sizes_t>(), sizes, cs, make_iterator_mock(), composite, int_t<1>(), tuple(42, 1));

            cudaMemcpy(outh, out.get(), 5 * 7 * 3 * 2 * 3 * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 5; ++i)
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 2; ++l) {
                        for (int m = 0; m < 3; ++m) {
                            int res = 42;
                            for (int j = 0; j < 7; ++j) {
                                res += inh[i][j][k][l][m];
                                EXPECT_EQ(outh[i][j][k][l][m], res);
                            }
                        }
                    }
                }
        }

        struct global_tmp_check_fun {
            template <class PtrHolder, class Strides>
            GT_FUNCTION bool operator()(PtrHolder ptr_holder, Strides strides) const {
                auto ptr = ptr_holder();
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 7; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            *ptr = 21 * i + 3 * j + k;
                            sid::shift(ptr, sid::get_stride<int_t<2>>(strides), 1_c);
                        }
                        sid::shift(ptr, sid::get_stride<int_t<2>>(strides), -3_c);
                        sid::shift(ptr, sid::get_stride<int_t<1>>(strides), 1_c);
                    }
                    sid::shift(ptr, sid::get_stride<int_t<1>>(strides), -7_c);
                    sid::shift(ptr, sid::get_stride<int_t<0>>(strides), 1_c);
                }
                sid::shift(ptr, sid::get_stride<int_t<0>>(strides), -5_c);
                bool correct = true;
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 7; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            correct &= *ptr == 21 * i + 3 * j + k;
                            sid::shift(ptr, sid::get_stride<int_t<2>>(strides), 1_c);
                        }
                        sid::shift(ptr, sid::get_stride<int_t<2>>(strides), -3_c);
                        sid::shift(ptr, sid::get_stride<int_t<1>>(strides), 1_c);
                    }
                    sid::shift(ptr, sid::get_stride<int_t<1>>(strides), -7_c);
                    sid::shift(ptr, sid::get_stride<int_t<0>>(strides), 1_c);
                }
                return correct;
            }
        };

        TEST(backend_gpu, global_tmp) {
            using block_sizes_t = meta::list<meta::list<int_t<0>, int_t<4>>, meta::list<int_t<2>, int_t<2>>>;
            auto alloc = tmp_allocator(gpu<block_sizes_t>());
            auto sizes = hymap::keys<int_t<0>, int_t<1>, int_t<2>>::values<int_t<5>, int_t<7>, int_t<3>>();
            auto tmp = allocate_global_tmp(alloc, sizes, data_type<int>());
            static_assert(sid::is_sid<decltype(tmp)>());
            auto ptr_holder = sid::get_origin(tmp);
            auto strides = sid::get_strides(tmp);
            bool success = on_device::exec(global_tmp_check_fun(), ptr_holder, strides);
            EXPECT_TRUE(success);
        }
    } // namespace
} // namespace gridtools::fn::backend
