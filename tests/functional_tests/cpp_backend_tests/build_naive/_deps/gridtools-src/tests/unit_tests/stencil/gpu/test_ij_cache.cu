/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/gpu/ij_cache.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/stencil/common/extent.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            using namespace literals;

            constexpr auto i_size = 3_c;
            constexpr auto j_size = 8_c;
            using extent_t = extent<-5, 1, -3, 1>;

            TEST(sid_ij_cache, smoke) {

                shared_allocator allocator;
                auto testee = make_ij_cache<double>(1_c, i_size, j_size, extent_t{}, allocator);

                using ij_cache_t = decltype(testee);

                static_assert(is_sid<ij_cache_t>());
                static_assert(std::is_same_v<sid::ptr_type<ij_cache_t>, double *>);
                static_assert(std::is_same_v<sid::ptr_diff_type<ij_cache_t>, int_t>);

                using expected_kind = hymap::keys<dim::c, dim::i, dim::j>::
                    values<integral_constant<int_t, 1>, integral_constant<int_t, 1>, integral_constant<int_t, 9>>;
                static_assert(std::is_same_v<sid::strides_kind<ij_cache_t>, expected_kind>);
                auto strides = sid::get_strides(testee);
                EXPECT_EQ(1, at_key<dim::i>(strides));
                EXPECT_EQ(9, at_key<dim::j>(strides));
                EXPECT_EQ(0, sid::get_stride<dim::k>(strides));
            }

            template <class IJCache1, class IJCache2, class Strides1, class Strides2>
            __device__ bool ij_cache_test(IJCache1 cache1, IJCache2 cache2, Strides1 strides1, Strides2 strides2) {

                auto ptr1 = cache1();
                auto ptr2 = cache2();

                // fill some data into the caches
                sid::shift(ptr1, sid::get_stride<dim::i>(strides1), extent_t::iminus());
                sid::shift(ptr1, sid::get_stride<dim::j>(strides1), extent_t::jminus());

                sid::shift(ptr2, sid::get_stride<dim::i>(strides2), extent_t::iminus());
                sid::shift(ptr2, sid::get_stride<dim::j>(strides2), extent_t::jminus());

                constexpr auto j_total_size = j_size - extent_t::jminus() + extent_t::jplus();
                constexpr auto i_total_size = i_size - extent_t::iminus() + extent_t::iplus();

                for (int j = 0; j < j_total_size; ++j) {
                    for (int i = 0; i < i_total_size; ++i) {
                        *ptr1 = 100 * j + i;
                        *ptr2 = 100 * j + i;

                        sid::shift(ptr1, sid::get_stride<dim::i>(strides1), 1_c);
                        sid::shift(ptr2, sid::get_stride<dim::i>(strides2), 1_c);
                    }
                    sid::shift(ptr1, sid::get_stride<dim::i>(strides1), -i_total_size);
                    sid::shift(ptr2, sid::get_stride<dim::i>(strides2), -i_total_size);

                    sid::shift(ptr1, sid::get_stride<dim::j>(strides1), 1_c);
                    sid::shift(ptr2, sid::get_stride<dim::j>(strides2), 1_c);
                }

                sid::shift(ptr1, sid::get_stride<dim::j>(strides1), -j_total_size);
                sid::shift(ptr2, sid::get_stride<dim::j>(strides2), -j_total_size);

                // shifting in k has no effect
                sid::shift(ptr1, sid::get_stride<dim::k>(strides1), 123);
                sid::shift(ptr2, sid::get_stride<dim::k>(strides2), 456);

                // verify that the data is still in there (no overwrites from the two caches)
                for (int j = 0; j < j_total_size; ++j) {
                    for (int i = 0; i < i_total_size; ++i) {
                        if (*ptr1 != 100. * j + i) {
                            printf(
                                "Float-Cache: Incorrect result at (i=%i, j=%i): %f != %f\n", i, j, *ptr1, 100. * j + i);
                            return false;
                        }
                        if (*ptr2 != 100 * j + i) {
                            printf("Int-Cache: Incorrect result at (i=%i, j=%i): %i != %i\n", i, j, *ptr2, 100 * j + i);
                            return false;
                        }
                        sid::shift(ptr1, sid::get_stride<dim::i>(strides1), 1_c);
                        sid::shift(ptr2, sid::get_stride<dim::i>(strides2), 1_c);
                    }
                    sid::shift(ptr1, sid::get_stride<dim::i>(strides1), -i_total_size);
                    sid::shift(ptr2, sid::get_stride<dim::i>(strides2), -i_total_size);

                    sid::shift(ptr1, sid::get_stride<dim::j>(strides1), 1_c);
                    sid::shift(ptr2, sid::get_stride<dim::j>(strides2), 1_c);
                }
                return true;
            }

            TEST(sid_ij_cache, sid) {
                shared_allocator allocator;
                auto cache1 = make_ij_cache<double>(1_c, i_size, j_size, extent_t{}, allocator);
                auto strides1 = sid::get_strides(cache1);
                auto ptr1 = sid::get_origin(cache1);

                auto cache2 = make_ij_cache<int16_t>(1_c, i_size, j_size, extent_t{}, allocator);
                auto strides2 = sid::get_strides(cache2);
                auto ptr2 = sid::get_origin(cache2);

                EXPECT_TRUE(gridtools::on_device::exec_with_shared_memory(allocator.size(),
                    GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(
                        (&ij_cache_test<decltype(ptr1), decltype(ptr2), decltype(strides1), decltype(strides2)>)),
                    ptr1,
                    ptr2,
                    strides1,
                    strides2));
            }
        } // namespace gpu_backend
    }     // namespace stencil
} // namespace gridtools
