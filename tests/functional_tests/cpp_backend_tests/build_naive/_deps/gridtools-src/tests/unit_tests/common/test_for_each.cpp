/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/for_each.hpp>

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/common/host_device.hpp>

namespace gridtools {

    struct f {
        int *&dst;

        template <class T>
        GT_FUNCTION_WARNING void operator()(T) const {
            *(dst++) = T::value;
        }
    };

    template <class...>
    struct lst;

    template <int I>
    using my_int_t = std::integral_constant<int, I>;

    TEST(for_each, empty) {
        int vals[3];
        int *cur = vals;
        for_each<lst<>>(f{cur});
        EXPECT_EQ(cur, cur);
    }

    TEST(for_each, functional) {
        int vals[3];
        int *cur = vals;
        for_each<lst<my_int_t<0>, my_int_t<42>, my_int_t<3>>>(f{cur});
        EXPECT_EQ(cur, vals + 3);
        EXPECT_THAT(vals, testing::ElementsAre(0, 42, 3));
    }

    namespace test_constexpr {
        struct f {
            int &dst;
            template <class T>
            constexpr void operator()(T) const {
                dst = T::value;
            }
        };
        constexpr int foo() {
            int res = 0;
            for_each<lst<my_int_t<2>>>(f{res});
            return res;
        }
        static_assert(foo() == 2);
    } // namespace test_constexpr

    TEST(for_each, targets) {
        int *ptr = nullptr;
        for_each<lst<>>(f{ptr});
        host::for_each<lst<>>(f{ptr});
        host_device::for_each<lst<>>(f{ptr});
    }
} // namespace gridtools
