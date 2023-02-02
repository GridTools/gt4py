/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/hymap.hpp>

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>

namespace gridtools {
    namespace {

        struct a;
        struct b;
        struct c;

        static_assert(!is_hymap<void>::value);
        static_assert(is_hymap<std::array<int, 3>>::value);
        static_assert(is_hymap<tuple<int, double>>::value);
        static_assert(is_hymap<hymap::keys<>::values<>>::value);
        static_assert(is_hymap<hymap::keys<a>::values<int>>::value);
        static_assert(is_hymap<hymap::keys<a, b>::values<int, double>>::value);

        static_assert(std::is_same_v<get_keys<std::array<int, 2>>,
            meta::list<integral_constant<int, 0>, integral_constant<int, 1>>>);
        static_assert(std::is_same_v<get_keys<tuple<int, double>>,
            meta::list<integral_constant<int, 0>, integral_constant<int, 1>>>);
        static_assert(std::is_same_v<get_keys<hymap::keys<>::values<>>, hymap::keys<>>);
        static_assert(std::is_same_v<get_keys<hymap::keys<a>::values<int>>, hymap::keys<a>>);

        TEST(tuple_like, smoke) {
            using testee_t = hymap::keys<a, b, c>::values<int, double, void *>;

            static_assert(tuple_util::size<testee_t>::value == 3);

            static_assert(std::is_same_v<tuple_util::element<0, testee_t>, int>);
            static_assert(std::is_same_v<tuple_util::element<1, testee_t>, double>);
            static_assert(std::is_same_v<tuple_util::element<2, testee_t>, void *>);

            testee_t testee{42, 5.3, nullptr};
            EXPECT_EQ(42, tuple_util::get<0>(testee));
            EXPECT_EQ(5.3, tuple_util::get<1>(testee));
            EXPECT_EQ(nullptr, tuple_util::get<2>(testee));
        }

        TEST(at_key, smoke) {
            using testee_t = hymap::keys<a, b>::values<int, double>;
            testee_t testee{42, 5.3};

            static_assert(has_key<testee_t, a>::value);
            static_assert(has_key<testee_t, b>::value);
            static_assert(!has_key<testee_t, c>::value);

            EXPECT_EQ(42, at_key<a>(testee));
            EXPECT_EQ(5.3, at_key<b>(testee));
        }

        TEST(at_key, tuple_like) {
            using testee_t = std::tuple<int, double>;
            testee_t testee{42, 5.3};

            static_assert(has_key<testee_t, integral_constant<int, 0>>::value);
            static_assert(has_key<testee_t, integral_constant<int, 1>>::value);
            static_assert(!has_key<testee_t, integral_constant<int, 2>>::value);

            EXPECT_EQ(42, (at_key<integral_constant<int, 0>>(testee)));
            EXPECT_EQ(5.3, (at_key<integral_constant<int, 1>>(testee)));
        }

        struct add_2_f {
            template <class T>
            T operator()(T x) const {
                return x + 2;
            }
        };

        TEST(tuple_like, transform) {
            using testee_t = hymap::keys<a, b>::values<int, double>;

            testee_t src = {42, 5.3};
            auto dst = tuple_util::transform(add_2_f{}, src);

            EXPECT_EQ(44, at_key<a>(dst));
            EXPECT_EQ(7.3, at_key<b>(dst));
        }

#if !defined(__NVCC__)
        TEST(deduction, smoke) {
            auto testee = hymap::keys<a, b>::values(42, 5.3);

            EXPECT_EQ(42, at_key<a>(testee));
            EXPECT_EQ(5.3, at_key<b>(testee));

            EXPECT_EQ(42, at_key<a>(hymap::keys<a>::values(42)));
        }
#if defined(__clang__) || defined(__GNUC__) && __GNUC__ > 8
        TEST(deduction, empty) { hymap::keys<>::values(); }
#endif
#endif

        TEST(convert_hymap, smoke) {
            auto src = hymap::keys<a, b>::make_values(42, 5.3);

            auto dst = tuple_util::convert_to<hymap::keys<b, c>::values>(src);

            EXPECT_EQ(42, at_key<b>(dst));
            EXPECT_EQ(5.3, at_key<c>(dst));
        }

        TEST(to_meta_map, smoke) {
            using src_t = hymap::keys<a, b>::values<int, double>;
            using dst_t = hymap::to_meta_map<src_t>;

            static_assert(std::is_same_v<meta::second<meta::mp_find<dst_t, a>>, int>);
            static_assert(std::is_same_v<meta::second<meta::mp_find<dst_t, b>>, double>);
        }

        TEST(from_meta_map, smoke) {
            using src_t = meta::list<meta::list<a, int>, meta::list<b, double>>;
            using dst_t = hymap::from_meta_map<src_t>;

            static_assert(std::is_same_v<typename std::decay<decltype(at_key<a>(std::declval<dst_t>()))>::type, int>);
            static_assert(
                std::is_same_v<typename std::decay<decltype(at_key<b>(std::declval<dst_t>()))>::type, double>);
        }

        TEST(from_meta_map, empty) {
            using src_t = meta::list<>;
            using dst_t = hymap::from_meta_map<src_t>;

            static_assert(tuple_util::size<dst_t>() == 0);
        }

        struct add_3_f {
            template <class Key, class Value>
            Value operator()(Value x) const {
                return x + 3;
            }
        };

        TEST(transform, smoke) {
            hymap::keys<a, b>::values<int, double> src = {42, 5.3};

            auto dst = hymap::transform(add_3_f{}, src);

            EXPECT_EQ(45, at_key<a>(dst));
            EXPECT_EQ(8.3, at_key<b>(dst));
        }

        struct acc_f {
            double &m_val;
            template <class Key, class Value>
            void operator()(Value x) const {
                m_val += x;
            }
        };

        TEST(for_each, smoke) {
            hymap::keys<a, b>::values<int, double> src = {42, 5.3};

            double acc = 0;
            hymap::for_each(acc_f{acc}, src);

            EXPECT_EQ(47.3, acc);
        }

        TEST(canonicalize_and_remove_key, smoke) {
            hymap::keys<a, b>::values<int, double> src = {42, 37.5};
            auto res = hymap::canonicalize_and_remove_key<a>(src);

            static_assert(!has_key<decltype(res), a>());
            static_assert(has_key<decltype(res), b>());

            EXPECT_EQ(37.5, at_key<b>(res));
        }

        TEST(merge, smoke) {
            auto m1 = hymap::keys<a, b>::make_values(1, 2);
            auto m2 = hymap::keys<b, c>::make_values(3.5, 16);
            auto testee = hymap::merge(m1, m2);

            EXPECT_EQ(1, at_key<a>(testee));
            EXPECT_EQ(2, at_key<b>(testee));
            EXPECT_EQ(16, at_key<c>(testee));
        }

        TEST(concat, smoke) {
            auto m1 = hymap::keys<a, b>::make_values(1, 2);
            auto m2 = hymap::keys<c>::make_values(3.5);
            auto testee = hymap::concat(m1, m2);

            EXPECT_EQ(1, at_key<a>(testee));
            EXPECT_EQ(2, at_key<b>(testee));
            EXPECT_EQ(3.5, at_key<c>(testee));
        }

        TEST(concat, empty) {
            auto m1 = hymap::keys<>::make_values();
            auto m2 = hymap::keys<a>::make_values(42);
            auto testee = hymap::concat(m1, m2);

            EXPECT_EQ(42, at_key<a>(testee));
        }

        TEST(assignment, smoke) {
            hymap::keys<a, b>::values<double, double> testee;
            auto src = hymap::keys<b, a, c>::make_values(88, 3.5, 16);
            testee = src;
            EXPECT_EQ(3.5, at_key<a>(testee));
            EXPECT_EQ(88, at_key<b>(testee));
        }

    } // namespace
} // namespace gridtools
