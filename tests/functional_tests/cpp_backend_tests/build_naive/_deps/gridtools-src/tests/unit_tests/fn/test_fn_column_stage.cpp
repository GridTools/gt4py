/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/column_stage.hpp>

#include <gtest/gtest.h>

#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        struct sum_fold : fwd {
            static GT_FUNCTION constexpr auto body() {
                return [](auto acc, auto const &iter) { return acc + *iter; };
            }
        };

        struct sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return tuple(get<0>(acc) + *iter, get<1>(acc) * *iter); },
                    [](auto acc) { return get<0>(acc); });
            }
        };

        struct sum_fold_with_logues : sum_fold {
            static GT_FUNCTION constexpr auto prologue() {
                return tuple([](auto acc, auto const &iter) { return acc + 2 * *iter; });
            }
            static GT_FUNCTION constexpr auto epilogue() {
                return tuple([](auto acc, auto const &iter) { return acc + 3 * *iter; });
            }
        };

        struct make_iterator_mock {
            auto GT_FUNCTION operator()() const {
                return [](auto tag, auto const &ptr, auto const & /*strides*/) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(scan, smoke) {
            using column_t = int[5];
            using vdim_t = integral_constant<int, 0>;

            column_t a = {0, 0, 0, 0, 0};
            column_t b = {1, 2, 3, 4, 5};
            auto composite = sid::composite::keys<integral_constant<int, 0>, integral_constant<int, 1>>::make_values(
                sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(&a[0]))
                    .set<property::strides>(tuple(1_c)),
                sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(&b[0]))
                    .set<property::strides>(tuple(1_c)));
            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);

            {
                column_stage<vdim_t, sum_fold, 0, 1> cs;
                auto res = cs(42, 5, make_iterator_mock()(), ptr, strides);
                EXPECT_EQ(res, 57);
                for (std::size_t i = 0; i < 5; ++i)
                    EXPECT_EQ(a[i], 0);
            }

            {
                column_stage<vdim_t, sum_scan, 0, 1> cs;
                auto res = cs(tuple(42, 1), 5, make_iterator_mock()(), ptr, strides);
                EXPECT_EQ(get<0>(res), 57);
                EXPECT_EQ(get<1>(res), 120);
                for (std::size_t i = 0; i < 5; ++i)
                    EXPECT_EQ(a[i], 42 + (i + 1) * (i + 2) / 2);
            }

            {
                column_stage<vdim_t, sum_fold_with_logues, 0, 1> cs;
                auto res = cs(42, 5, make_iterator_mock()(), ptr, strides);
                EXPECT_EQ(res, 68);
            }

            {
                merged_column_stage<column_stage<vdim_t, sum_scan, 0, 1>, column_stage<vdim_t, sum_scan, 0, 1>> cs;
                auto res = cs(tuple(0, 1), 5, make_iterator_mock()(), ptr, strides);
                EXPECT_EQ(get<0>(res), 2 * 15);
                EXPECT_EQ(get<1>(res), 120 * 120);
                for (std::size_t i = 0; i < 5; ++i)
                    EXPECT_EQ(a[i], 15 + (i + 1) * (i + 2) / 2);
            }
        }

    } // namespace
} // namespace gridtools::fn
