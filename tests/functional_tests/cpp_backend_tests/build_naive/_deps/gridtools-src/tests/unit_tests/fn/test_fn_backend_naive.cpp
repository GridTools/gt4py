/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/backend/naive.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/column_stage.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

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
            auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(backend_naive, apply_column_stage) {
            int in[5][7][3], out[5][7][3] = {};
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 7; ++j)
                    for (int k = 0; k < 3; ++k)
                        in[i][j][k] = 21 * i + 3 * j + k;

            auto as_synthetic = [](int x[5][7][3]) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(&x[0][0][0]))
                    .set<property::strides>(tuple(21_c, 3_c, 1_c));
            };

            auto composite = sid::composite::keys<int_t<0>, int_t<1>>::make_values(as_synthetic(out), as_synthetic(in));

            auto sizes = hymap::keys<int_t<0>, int_t<1>, int_t<2>>::values<int_t<5>, int_t<7>, int_t<3>>();

            column_stage<int_t<1>, sum_scan, 0, 1> cs;

            apply_column_stage(naive(), sizes, cs, make_iterator_mock(), composite, int_t<1>(), tuple(42, 1));

            for (int i = 0; i < 5; ++i)
                for (int k = 0; k < 3; ++k) {
                    int res = 42;
                    for (int j = 0; j < 7; ++j) {
                        res += in[i][j][k];
                        EXPECT_EQ(out[i][j][k], res);
                    }
                }
        }

        TEST(backend_naive, global_tmp) {
            auto alloc = tmp_allocator(naive());
            auto sizes = hymap::keys<int_t<0>, int_t<1>, int_t<2>>::values<int_t<5>, int_t<7>, int_t<3>>();
            auto tmp = allocate_global_tmp(alloc, sizes, data_type<int>());
            static_assert(sid::is_sid<decltype(tmp)>());

            auto ptr = sid::get_origin(tmp)();
            auto strides = sid::get_strides(tmp);
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

            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 7; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        EXPECT_EQ(*ptr, 21 * i + 3 * j + k);
                        sid::shift(ptr, sid::get_stride<int_t<2>>(strides), 1_c);
                    }
                    sid::shift(ptr, sid::get_stride<int_t<2>>(strides), -3_c);
                    sid::shift(ptr, sid::get_stride<int_t<1>>(strides), 1_c);
                }
                sid::shift(ptr, sid::get_stride<int_t<1>>(strides), -7_c);
                sid::shift(ptr, sid::get_stride<int_t<0>>(strides), 1_c);
            }
            sid::shift(ptr, sid::get_stride<int_t<0>>(strides), -5_c);
        }
    } // namespace
} // namespace gridtools::fn::backend
