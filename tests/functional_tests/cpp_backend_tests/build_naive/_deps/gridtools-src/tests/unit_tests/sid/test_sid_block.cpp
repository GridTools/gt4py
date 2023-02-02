/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/block.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>
#include <gridtools/stencil/common/dim.hpp>
#include <gridtools/stencil/positional.hpp>

namespace gridtools {
    namespace {
        using namespace stencil;
        struct some_dim;

        using positional_t = sid::composite::keys<dim::i, dim::j, dim::k>::
            values<positional<dim::i>, positional<dim::j>, positional<dim::k>>;

        TEST(sid_block, smoke) {
            const int domain_size_i = 12;
            const int domain_size_j = 14;
            const int domain_size_k = 4;
            constexpr int block_size_i = 3;
            const int block_size_j = 7;

            auto blocks = hymap::keys<dim::i, dim::j, some_dim>::make_values(
                integral_constant<int_t, block_size_i>{}, block_size_j, 5);

            positional_t s;
            auto blocked_s = sid::block(s, blocks);
            static_assert(is_sid<decltype(blocked_s)>());

            auto strides = sid::get_strides(blocked_s);
            for (int ib = 0; ib < domain_size_i; ib += block_size_i) {
                for (int jb = 0; jb < domain_size_j; jb += block_size_j) {
                    auto ptr = sid::get_origin(blocked_s)();
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), ib / block_size_i);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::j>>(strides), jb / block_size_j);

                    for (int i = ib; i < ib + block_size_i; ++i) {
                        for (int j = jb; j < jb + block_size_j; ++j) {
                            for (int k = 0; k < domain_size_k; ++k) {
                                EXPECT_EQ(*at_key<dim::i>(ptr), i);
                                EXPECT_EQ(*at_key<dim::j>(ptr), j);
                                EXPECT_EQ(*at_key<dim::k>(ptr), k);
                                sid::shift(ptr, sid::get_stride<dim::k>(strides), 1);
                            }
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), -domain_size_k);
                            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1);
                        }
                        sid::shift(ptr, sid::get_stride<dim::j>(strides), -block_size_j);
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), 1);
                    }
                }
            }
        }

        TEST(sid_block, multilevel) {
            positional_t s;

            const int domain_size = 20;
            const int block_size_1 = 5;
            const int block_size_2 = 2;
            auto blocked_s = sid::block(s, hymap::keys<dim::i>::make_values(block_size_1));
            static_assert(is_sid<decltype(blocked_s)>());
            auto blocked_blocked_s =
                sid::block(blocked_s, hymap::keys<sid::blocked_dim<dim::i>>::make_values(block_size_2));
            static_assert(is_sid<decltype(blocked_blocked_s)>());

            auto ptr = sid::get_origin(blocked_blocked_s)();
            auto strides = sid::get_strides(blocked_blocked_s);
            for (int ib2 = 0; ib2 < domain_size; ib2 += block_size_1 * block_size_2) {
                for (int ib = ib2; ib < ib2 + block_size_1 * block_size_2; ib += block_size_1) {
                    for (int i = ib; i < ib + block_size_1; ++i) {
                        EXPECT_EQ(*at_key<dim::i>(ptr), i);
                        EXPECT_EQ(*at_key<dim::j>(ptr), 0);
                        EXPECT_EQ(*at_key<dim::k>(ptr), 0);
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), 1);
                    }
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), -block_size_1);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), 1);
                }
                sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), -block_size_2);
                sid::shift(ptr, sid::get_stride<sid::blocked_dim<sid::blocked_dim<dim::i>>>(strides), 1);
            }
        }

        TEST(sid_block, do_nothing) {
            positional_t s;

            auto same_s = sid::block(s, hymap::keys<some_dim>::make_values(42));
            static_assert(std::is_same_v<decltype(s), decltype(same_s)>);
        }

        TEST(sid_block, reference_wrapper) {
            positional_t s;

            const int domain_size = 20;
            const int block_size = 5;
            auto blocked_s = sid::block(std::ref(s), hymap::keys<dim::i>::make_values(block_size));
            static_assert(is_sid<decltype(blocked_s)>());

            auto ptr = sid::get_origin(blocked_s)();
            auto strides = sid::get_strides(blocked_s);
            for (int ib = 0; ib < domain_size; ib += block_size) {
                for (int i = ib; i < ib + block_size; ++i) {
                    EXPECT_EQ(*at_key<dim::i>(ptr), i);
                    EXPECT_EQ(*at_key<dim::j>(ptr), 0);
                    EXPECT_EQ(*at_key<dim::k>(ptr), 0);
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), 1);
                }
                sid::shift(ptr, sid::get_stride<dim::i>(strides), -block_size);
                sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), 1);
            }
        }

        TEST(sid_block, integral_strides_types) {
            using namespace literals;

            using dims_t = hymap::keys<dim::i, dim::j, dim::k, some_dim>;

            auto strides = dims_t::make_values(1, 2, 4_c, 8_c);
            auto blocks = dims_t::make_values(2, 2_c, 2, 2_c);

            auto s = sid::synthetic()
                         .set<sid::property::origin>(sid::simple_ptr_holder<int *>{nullptr})
                         .set<sid::property::strides>(strides)
                         .set<sid::property::strides_kind, void>();
            static_assert(is_sid<decltype(s)>());

            auto blocked_s = sid::block(s, blocks);
            static_assert(is_sid<decltype(blocked_s)>());

            auto blocked_strides = sid::get_strides(blocked_s);
            EXPECT_EQ(sid::get_stride<sid::blocked_dim<dim::i>>(blocked_strides), 2);
            EXPECT_EQ(sid::get_stride<sid::blocked_dim<dim::j>>(blocked_strides), 4);
            EXPECT_EQ(sid::get_stride<sid::blocked_dim<dim::k>>(blocked_strides), 8);
            EXPECT_EQ(sid::get_stride<sid::blocked_dim<some_dim>>(blocked_strides), 16_c);
        }
    } // namespace
} // namespace gridtools
