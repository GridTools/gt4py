/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/concept.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/simple_ptr_holder.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        // several primitive not sids
        static_assert(!is_sid<void>());
        static_assert(!is_sid<int>());
        struct garbage {};
        static_assert(!is_sid<garbage>());

        // fully custom defined sid
        namespace custom {
            struct element {};
            struct ptr_diff {
                int val;
            };
            struct ptr {
                element *val;
                GT_FUNCTION element &operator*() const { return *val; }
                friend GT_FUNCTION ptr operator+(ptr, ptr_diff) { return {}; }
            };
            struct stride {
                friend GT_FUNCTION std::true_type sid_shift(ptr &, stride const &, int) { return {}; }
                friend GT_FUNCTION std::false_type sid_shift(ptr_diff &, stride const &, int) { return {}; }
            };

            struct dim_0;
            struct dim_1;

            using strides = hymap::keys<dim_0, dim_1>::values<stride, stride>;

            struct strides_kind;
            struct bounds_validator_kind;

            struct testee {
                friend sid::host_device::simple_ptr_holder<ptr> sid_get_origin(testee &) { return {}; }
                friend strides sid_get_strides(testee const &) { return {}; }

                friend ptr_diff sid_get_ptr_diff(testee);
                friend strides_kind sid_get_strides_kind(testee);
                friend bounds_validator_kind sid_get_bounds_validator_kind(testee);
            };

            static_assert(sid::concept_impl_::is_sid<testee>());
            static_assert(std::is_same_v<sid::ptr_diff_type<testee>, ptr_diff>);
            static_assert(std::is_same_v<sid::reference_type<testee>, element &>);
            static_assert(std::is_same_v<sid::element_type<testee>, element>);
            static_assert(std::is_same_v<sid::const_reference_type<testee>, element const &>);
            static_assert(std::is_same_v<sid::strides_kind<testee>, strides_kind>);

            static_assert(std::is_same_v<std::decay_t<decltype(sid::get_origin(std::declval<testee &>())())>, ptr>);
            static_assert(std::is_same_v<decltype(sid::get_strides(testee{})), strides>);

            static_assert(std::is_same_v<std::decay_t<decltype(sid::get_stride<dim_0>(strides{}))>, stride>);
            static_assert(std::is_same_v<std::decay_t<decltype(sid::get_stride<dim_1>(strides{}))>, stride>);
            static_assert(decltype(sid::get_stride<void>(strides()))::value == 0);
            static_assert(decltype(sid::get_stride<void *>(strides()))::value == 0);

            static_assert(std::is_same_v<decltype(sid::shift(std::declval<ptr &>(), stride{}, 0)), std::true_type>);
            static_assert(
                std::is_same_v<decltype(sid::shift(std::declval<ptr_diff &>(), stride{}, 0)), std::false_type>);
        } // namespace custom

        namespace fallbacks {

            struct testee {
                friend sid::host_device::simple_ptr_holder<testee *> sid_get_origin(testee &obj) { return {&obj}; }
            };

            static_assert(is_sid<testee>());
            static_assert(std::is_same_v<sid::ptr_type<testee>, testee *>);
            static_assert(std::is_same_v<sid::ptr_diff_type<testee>, ptrdiff_t>);
            static_assert(std::is_same_v<sid::reference_type<testee>, testee &>);
            static_assert(std::is_same_v<sid::element_type<testee>, testee>);
            static_assert(std::is_same_v<sid::const_reference_type<testee>, testee const &>);

            using strides = sid::strides_type<testee>;
            static_assert(tuple_util::size<strides>() == 0);

            static_assert(
                std::is_same_v<std::decay_t<decltype(sid::get_origin(std::declval<testee &>())())>, testee *>);
            static_assert(std::is_same_v<decltype(sid::get_strides(testee{})), strides>);

            auto stride = sid::get_stride<void>(strides{});
            static_assert(decltype(stride)::value == 0);

            static_assert(std::is_void_v<std::void_t<decltype(sid::shift(std::declval<testee *&>(), stride, 42))>>);
            static_assert(std::is_void_v<std::void_t<decltype(sid::shift(std::declval<ptrdiff_t *&>(), stride, 42))>>);

            using lower_bounds = sid::lower_bounds_type<testee>;
            static_assert(tuple_util::size<lower_bounds>() == 0);

            using upper_bounds = sid::upper_bounds_type<testee>;
            static_assert(tuple_util::size<upper_bounds>() == 0);
        } // namespace fallbacks

        template <class T, class Stride, class Offset>
        void do_verify_shift(T obj, Stride stride, Offset offset) {
            auto expected = obj + stride * offset;
            sid::shift(obj, stride, offset);
            EXPECT_EQ(expected, obj);
        }

        struct verify_shift_f {
            template <class Stride, class Offset>
            void operator()(Stride stride, Offset offset) const {
                int const data[100] = {};
                do_verify_shift(data + 50, stride, offset);
                do_verify_shift(42, stride, offset);
            }
        };

        TEST(shift, default_overloads) {
            auto samples = tuple(2, 3, -2_c, -1_c, 0_c, 1_c, 2_c);
            tuple_util::for_each_in_cartesian_product(verify_shift_f{}, samples, samples);
        }

        TEST(shift, noop) {
            // this should compile
            sid::shift(3, 4, 5);
        }

        namespace non_static_value {
            struct stride {
                int value;
            };

            struct testee {};

            sid::host_device::simple_ptr_holder<testee *> sid_get_origin(testee &obj) { return {&obj}; }
            tuple<stride> sid_get_strides(testee const &) { return {}; }
            GT_FUNCTION int operator*(stride, int) { return 100; }
            integral_constant<int, 42> sid_get_strides_kind(testee const &);

            static_assert(is_sid<testee>());

            TEST(non_static_value, shift) {
                ptrdiff_t val = 22;
                sid::shift(val, stride{}, 0);
                EXPECT_EQ(122, val);
                val = 22;
                sid::shift(val, stride{}, 3_c);
                EXPECT_EQ(122, val);
            }
        } // namespace non_static_value

        TEST(c_array, smoke) {
            double testee[15][43] = {};
            static_assert(is_sid<decltype(testee)>());

            EXPECT_EQ(&testee[0][0], sid::get_origin(testee)());

            auto strides = sid::get_strides(testee);
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 0>>(strides) == 43));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 1>>(strides) == 1));

            using strides_t = decltype(strides);

            static_assert(tuple_util::size<strides_t>::value == 2);

            using stride_0_t = tuple_util::element<0, strides_t>;
            using stride_1_t = tuple_util::element<1, strides_t>;

            static_assert(stride_0_t::value == 43);
            static_assert(stride_1_t::value == 1);

            testee[7][8] = 555;

            auto *ptr = sid::get_origin(testee)();
            sid::shift(ptr, sid::get_stride<integral_constant<int, 0>>(strides), 7);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 1>>(strides), 8);

            EXPECT_EQ(555, *ptr);

            auto lower_bounds = sid::get_lower_bounds(testee);
            EXPECT_EQ(0, tuple_util::get<0>(lower_bounds));
            EXPECT_EQ(0, tuple_util::get<1>(lower_bounds));

            auto upper_bounds = sid::get_upper_bounds(testee);
            EXPECT_EQ(15, tuple_util::get<0>(upper_bounds));
            EXPECT_EQ(43, tuple_util::get<1>(upper_bounds));
        }

        TEST(c_array, 4D) {
            double testee[2][3][4][5] = {};
            static_assert(is_sid<decltype(testee)>());

            EXPECT_EQ(&testee[0][0][0][0], sid::get_origin(testee)());

            auto strides = sid::get_strides(testee);
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 0>>(strides) == 60));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 1>>(strides) == 20));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 2>>(strides) == 5));
            EXPECT_TRUE((sid::get_stride<integral_constant<int, 3>>(strides) == 1));

            testee[1][2][3][4] = 555;

            auto *ptr = sid::get_origin(testee)();
            sid::shift(ptr, sid::get_stride<integral_constant<int, 0>>(strides), 1);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 1>>(strides), 2);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 2>>(strides), 3);
            sid::shift(ptr, sid::get_stride<integral_constant<int, 3>>(strides), 4);

            EXPECT_EQ(555, *ptr);

            auto lower_bounds = sid::get_lower_bounds(testee);
            EXPECT_EQ(0, tuple_util::get<0>(lower_bounds));
            EXPECT_EQ(0, tuple_util::get<1>(lower_bounds));
            EXPECT_EQ(0, tuple_util::get<2>(lower_bounds));
            EXPECT_EQ(0, tuple_util::get<3>(lower_bounds));

            auto upper_bounds = sid::get_upper_bounds(testee);
            EXPECT_EQ(2, tuple_util::get<0>(upper_bounds));
            EXPECT_EQ(3, tuple_util::get<1>(upper_bounds));
            EXPECT_EQ(4, tuple_util::get<2>(upper_bounds));
            EXPECT_EQ(5, tuple_util::get<3>(upper_bounds));
        }

#ifdef __cpp_concepts
        namespace cpp20_concept {
            std::false_type foo(...);
            std::true_type foo(Sid auto const &);

            int bad = 42;
            int good[] = {42};

            static_assert(!decltype(foo(bad))());
            static_assert(decltype(foo(good))());
        } // namespace cpp20_concept
#endif

    } // namespace
} // namespace gridtools
