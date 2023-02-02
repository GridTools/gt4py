/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/executor.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/backend/naive.hpp>
#include <gridtools/fn/column_stage.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        template <int I>
        using int_t = integral_constant<int, I>;

        struct stencil {
            GT_FUNCTION constexpr auto operator()() const {
                return [](auto const &iter) { return 2 * *iter; };
            }
        };

        struct fwd_sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass([](auto acc, auto const &iter) { return acc + *iter; }, [](auto acc) { return acc; });
            }
        };

        struct bwd_sum_scan : bwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass([](auto acc, auto const &iter) { return acc + *iter; }, [](auto acc) { return acc; });
            }
        };

        struct make_iterator_mock {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(stencil_executor, smoke) {
            using backend_t = backend::naive;
            auto domain = hymap::keys<int_t<0>, int_t<1>>::make_values(2_c, 3_c);

            auto alloc = tmp_allocator(backend_t());
            int a[2][3] = {}, b[2][3] = {}, c[2][3];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    c[i][j] = 3 * i + j;

            make_stencil_executor(backend_t(), domain, std::tuple<>(), make_iterator_mock())
                .arg(a)
                .arg(b)
                .arg(c)
                .assign(1_c, stencil(), 2_c)
                .assign(0_c, stencil(), 1_c)
                .execute();

            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j) {
                    EXPECT_EQ(a[i][j], (3 * i + j) * 4);
                    EXPECT_EQ(b[i][j], (3 * i + j) * 2);
                    EXPECT_EQ(c[i][j], (3 * i + j) * 1);
                }
        }

        TEST(vertical_executor, smoke) {
            using backend_t = backend::naive;
            auto domain = hymap::keys<int_t<0>, int_t<1>>::make_values(2_c, 3_c);

            int a[2][3] = {}, b[2][3] = {}, c[2][3];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    c[i][j] = 3 * i + j;

            make_vertical_executor<int_t<1>>(backend_t(), domain, std::tuple<>(), make_iterator_mock())
                .arg(a)
                .arg(b)
                .arg(c)
                .assign(1_c, fwd_sum_scan(), 42, 2_c)
                .assign(0_c, bwd_sum_scan(), 8, 1_c)
                .execute();

            for (int i = 0; i < 2; ++i) {
                int res = 42;
                for (int j = 0; j < 3; ++j) {
                    EXPECT_EQ(c[i][j], 3 * i + j);
                    res += c[i][j];
                    EXPECT_EQ(b[i][j], res);
                }
                res = 8;
                for (int j = 2; j >= 0; --j) {
                    res += b[i][j];
                    EXPECT_EQ(a[i][j], res);
                }
            }
        }
    } // namespace
} // namespace gridtools::fn
