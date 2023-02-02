/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/tuple_util.hpp>

#include <array>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/pair.hpp>
#include <gridtools/meta.hpp>

namespace custom {
    struct foo {
        GT_STRUCT_TUPLE(foo, (int, a), (double, b));
    };
} // namespace custom

namespace gridtools {

    static_assert(!is_tuple_like<void>::value);
    static_assert(!is_tuple_like<int>::value);
    static_assert(is_tuple_like<std::array<int, 42>>::value);
    static_assert(is_tuple_like<std::tuple<>>::value);
    static_assert(is_tuple_like<std::tuple<int>>::value);
    static_assert(is_tuple_like<std::pair<int, double>>::value);
    static_assert(is_tuple_like<array<int, 42>>::value);
    static_assert(is_tuple_like<pair<int, double>>::value);
    static_assert(is_tuple_like<custom::foo>::value);
    static_assert(is_tuple_like<std::tuple<int, int const, int &&, int &, int const &>>::value);

#ifdef __cpp_concepts
    static_assert(!concepts::tuple_like<int>);
    static_assert(concepts::tuple_like<std::tuple<>>);
    static_assert(concepts::tuple_like_of<std::tuple<>>);
    static_assert(concepts::tuple_like_of<std::tuple<int, double>, int, double>);
    static_assert(!concepts::tuple_like_of<std::tuple<int, double, double>, int, double>);
#endif

    namespace tuple_util {
        TEST(get, std_tuple) {
            auto obj = std::tuple(1, 2.);
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);
        }

        TEST(get, std_pair) {
            auto obj = std::pair(1, 2.);
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);
        }

        TEST(get, std_array) {
            auto obj = std::array{1, 2};
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);
        }

        struct add_2_f {
            template <class T>
            GT_FUNCTION constexpr T operator()(T val) const {
                return val + 2;
            }
        };

        TEST(get, custom) {
            custom::foo obj{1, 2};
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);

            EXPECT_EQ(size<custom::foo>::value, 2);

            auto res = transform(add_2_f{}, custom::foo{42, 5.3});
            static_assert(std::is_same_v<decltype(res), custom::foo>);
            EXPECT_EQ(res.a, 44);
            EXPECT_EQ(res.b, 7.3);
        }

        TEST(transform, functional) {
            auto src = std::tuple(42, 5.3);
            auto res = transform(add_2_f{}, src);
            static_assert(std::is_same_v<decltype(res), decltype(src)>);
            EXPECT_EQ(res, std::tuple(44, 7.3));
        }

        TEST(transform, array) {
            auto src = std::array{42, 5};
            auto res = transform(add_2_f{}, src);
            static_assert(std::is_same_v<decltype(res), decltype(src)>);
            EXPECT_THAT(res, testing::ElementsAre(44, 7));
        }

        TEST(transform, gt_array) {
            auto src = gridtools::array{42, 5};
            auto res = host_device::transform(add_2_f{}, src);
            static_assert(std::is_same_v<decltype(res), decltype(src)>);
            EXPECT_THAT(res, testing::ElementsAre(44, 7));
        }

        TEST(transform, multiple_inputs) {
            EXPECT_EQ(std::tuple(11, 22),
                transform([](int lhs, int rhs) { return lhs + rhs; }, std::tuple(1, 2), std::tuple(10, 20)));
        }

        TEST(transform, multiple_arrays) {
            EXPECT_THAT(transform([](int lhs, int rhs) { return lhs + rhs; }, std::array{1, 2}, std::array{10, 20}),
                testing::ElementsAre(11, 22));
        }

        struct add_index_f {
            template <size_t I, class T>
            GT_FUNCTION constexpr T operator()(T val) const {
                return val + I;
            }
        };

        TEST(transform_index, functional) {
            auto src = std::tuple(42, 5.3);
            auto res = transform_index(add_index_f{}, src);
            static_assert(std::is_same_v<decltype(res), decltype(src)>);
            EXPECT_EQ(res, std::tuple(42, 6.3));
        }

        struct accumulate_f {
            double &m_acc;
            template <class T>
            void operator()(T val) const {
                m_acc += val;
            }
        };

        TEST(for_each, functional) {
            double acc = 0;
            for_each(accumulate_f{acc}, std::tuple(42, 5.3));
            EXPECT_EQ(47.3, acc);
        }

        TEST(for_each, array) {
            double acc = 0;
            for_each(accumulate_f{acc}, std::array{42., 5.3});
            EXPECT_EQ(47.3, acc);
        }

        struct accumulate_index_f {
            int &m_acc;
            template <size_t I, class T>
            void operator()(T val) const {
                m_acc += (I + 1) * val;
            }
        };

        TEST(for_each_index, functional) {
            int acc = 0;
            for_each_index(accumulate_index_f{acc}, std::tuple(42, 5));
            EXPECT_EQ(52, acc);
        }

        TEST(for_each, multiple_inputs) {
            int acc = 0;
            for_each([&](int lhs, int rhs) { acc += lhs + rhs; }, std::tuple(1, 2), std::tuple(10, 20));
            EXPECT_EQ(33, acc);
        }

        struct accumulate2_f {
            double &m_acc;
            template <class T, class U>
            void operator()(T lhs, U rhs) const {
                m_acc += lhs * rhs;
            }
        };

        TEST(for_each_in_cartesian_product, functional) {
            double acc = 0;
            for_each_in_cartesian_product(accumulate2_f{acc}, std::tuple(1, 2), std::tuple(10, 20));
            EXPECT_EQ(90, acc);
        }

        TEST(for_each_in_cartesian_product, array) {
            double acc = 0;
            for_each_in_cartesian_product(accumulate2_f{acc}, std::array{1, 2}, std::array{10, 20});
            EXPECT_EQ(90, acc);
        }

        TEST(concat, functional) { EXPECT_EQ(concat(std::tuple(1, 2), std::tuple(3, 4)), std::tuple(1, 2, 3, 4)); }

        TEST(concat, mixed) { EXPECT_EQ(concat(std::tuple<>(), std::array{3, 4}), std::tuple(3, 4)); }

        TEST(concat, array) {
            EXPECT_THAT(concat(std::array{1, 2}, std::array{3, 4}), testing::ElementsAre(1, 2, 3, 4));
        }

        TEST(flatten, functional) {
            EXPECT_EQ(flatten(std::tuple(std::tuple(1, 2), std::tuple(3, 4))), std::tuple(1, 2, 3, 4));
        }

        TEST(flatten, array) {
            EXPECT_THAT(flatten(std::tuple(std::array{1, 2}, std::array{3, 4})), testing::ElementsAre(1, 2, 3, 4));
        }

        TEST(flatten, ref) {
            auto orig = std::tuple(std::tuple(1, 2), std::tuple(3, 4));
            auto flat = flatten(orig);
            EXPECT_EQ(flat, std::tuple(1, 2, 3, 4));
            get<0>(flat) = 42;
            EXPECT_EQ(get<0>(get<0>(orig)), 42);
        }

        TEST(drop_front, functional) { EXPECT_EQ(drop_front<2>(std::tuple(1, 2, 3, 4)), std::tuple(3, 4)); }

        TEST(drop_front, array) { EXPECT_THAT(drop_front<2>(std::array{1, 2, 3, 4}), testing::ElementsAre(3, 4)); }

        TEST(push_back, functional) { EXPECT_EQ(push_back(std::tuple(1, 2), 3, 4), std::tuple(1, 2, 3, 4)); }

        TEST(push_back, array) { EXPECT_THAT(push_back(std::array{1, 2}, 3, 4), testing::ElementsAre(1, 2, 3, 4)); }

        TEST(push_front, functional) { EXPECT_EQ(push_front(std::tuple(1, 2), 3, 4), std::tuple(3, 4, 1, 2)); }

        TEST(pop_back, functional) { EXPECT_EQ(pop_back(std::tuple(1, 2, 3, 4)), std::tuple(1, 2, 3)); }

        TEST(pop_front, functional) { EXPECT_EQ(pop_front(std::tuple(1, 2, 3, 4)), std::tuple(2, 3, 4)); }

        TEST(fold, functional) {
            auto f = [](int x, int y) { return x + y; };
            EXPECT_EQ(fold(f, std::tuple(1, 2, 3, 4, 5, 6)), 21);
        }

        TEST(fold, with_state) {
            auto f = [](int x, int y) { return x + y; };
            EXPECT_EQ(fold(f, 100, std::tuple(1, 2, 3, 4, 5, 6)), 121);
        }

        TEST(fold, array) {
            auto f = [](int x, int y) { return x + y; };
            EXPECT_EQ(fold(f, std::array{1, 2, 3, 4, 5, 6}), 21);
        }

        TEST(apply, lambda) {
            auto f = [](int x, int y) { return x + y; };
            auto t = std::tuple(1, 2);

            EXPECT_EQ(
                3, gridtools::tuple_util::apply(f, t)); // fully qualified to resolve ambiguity with c++17 std::apply
        }

        TEST(apply, array) {
            EXPECT_EQ(3,
                gridtools::tuple_util::apply([](int x, int y) { return x + y; },
                    std::array{1, 2})); // fully qualified to resolve ambiguity with c++17 std::apply
        }

        TEST(tie, functional) {
            int x = 0, y = 0;
            tie<std::tuple>(x, y) = std::tuple(3, 4);
            EXPECT_EQ(3, x);
            EXPECT_EQ(4, y);
        }

        TEST(tie, pair) {
            int x = 0, y = 0;
            tie<std::pair>(x, y) = std::pair(3, 4);
            EXPECT_EQ(3, x);
            EXPECT_EQ(4, y);
        }

        TEST(convert_to, tuple) {
            EXPECT_EQ(std::tuple(1, 2), convert_to<std::tuple>(std::array{1, 2}));
            EXPECT_EQ(std::tuple(1, 2), convert_to<std::tuple>(gridtools::pair(1, 2)));
            EXPECT_EQ(std::pair(1, 2), convert_to<std::pair>(gridtools::array{1, 2}));
        }

        TEST(convert_to, array) {
            EXPECT_THAT(convert_to<std::array>()(std::tuple(3, 4)), testing::ElementsAre(3, 4));
            EXPECT_THAT(convert_to<std::array>(std::tuple(3, 4)), testing::ElementsAre(3, 4));
            EXPECT_THAT(convert_to<std::array>(std::tuple(3.5, 4)), testing::ElementsAre(3.5, 4.));
            EXPECT_THAT((convert_to<std::array, int>(std::tuple(3.5, 4))), testing::ElementsAre(3, 4));
            EXPECT_THAT((convert_to<std::array, int>()(std::tuple(3.5, 4))), testing::ElementsAre(3, 4));
            EXPECT_THAT((convert_to<std::array, int>(std::array{3.5, 4.})), testing::ElementsAre(3, 4));
        }

        TEST(transpose, functional) {
            EXPECT_EQ(transpose(std::array{std::pair(1, 10), std::pair(2, 20), std::pair(3, 30)}),
                std::pair(std::array{1, 2, 3}, std::array{10, 20, 30}));
        }

        TEST(all_of, functional) {
            auto testee = all_of([](int i) { return i % 2; });
            EXPECT_TRUE(testee(std::tuple(1, 3, 99, 7)));
            EXPECT_FALSE(testee(std::tuple(1, 3, 2, 7, 100)));

            EXPECT_TRUE(all_of([](int l, int r) { return l == r; }, std::tuple(1, 3, 99, 7), std::tuple(1, 3, 99, 7)));
        }

        TEST(reverse, functional) { EXPECT_TRUE(reverse(std::tuple(1, 'a', 42.5)) == std::tuple(42.5, 'a', 1)); }

        TEST(insert, functional) { EXPECT_TRUE(insert<2>('a', std::tuple(1, 2, 3)) == std::tuple(1, 2, 'a', 3)); }
    } // namespace tuple_util
} // namespace gridtools
