/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/ct_dispatch.hpp>

#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        TEST(ct_dispatch, smoke) {
            auto testee = [](size_t n) { return ct_dispatch<10>([](auto n) { return decltype(n)::value; }, n); };
            for (size_t i = 0; i != 10; ++i)
                EXPECT_EQ(testee(i), i);
        }

        int counter = 0;

        int count() {
            ++counter;
            return 0;
        }

        template <size_t>
        struct foo {
            static int dummy;
        };

        template <size_t I>
        int foo<I>::dummy = count();

        void bar(int n) {
            ct_dispatch<42>([](auto n) { (void)foo<decltype(n)::value>::dummy; }, n);
        }

        // check that foo is instantiated 42 times without even calling bar.
        TEST(ct_dispatch, instantiations_number) { EXPECT_EQ(counter, 42); }

    } // namespace
} // namespace gridtools
