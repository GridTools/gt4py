/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/traits.hpp>

#include <storage_select.hpp>

using namespace gridtools;

template <class T, int... Args>
void run() {
    constexpr int h1 = 3;
    constexpr int h2 = 4;
    constexpr int h3 = 5;
    auto store = storage::builder<storage_traits_t>
            .template type<T>()
            .template layout<Args...>()
            .dimensions(12, 34, 12)
            .halos(h1, h2, h3)
            .build();

    auto view = store->target_view();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(store->get_target_ptr() + store->info().index(h1, h2, h3)) %
                  storage::traits::byte_alignment<storage_traits_t>,
        0);
}

TEST(Storage, InnerRegionAlignmentChar210) { run<char, 2, 1, 0>(); }

TEST(Storage, InnerRegionAlignmentInt210) { run<int, 2, 1, 0>(); }

TEST(Storage, InnerRegionAlignmentFloat210) { run<float, 2, 1, 0>(); }

TEST(Storage, InnerRegionAlignmentDouble210) { run<double, 2, 1, 0>(); }

TEST(Storage, InnerRegionAlignmentChar012) { run<char, 0, 1, 2>(); }

TEST(Storage, InnerRegionAlignmentInt012) { run<int, 0, 1, 2>(); }

TEST(Storage, InnerRegionAlignmentFloat012) { run<float, 0, 1, 2>(); }

TEST(Storage, InnerRegionAlignmentDouble012) { run<double, 0, 1, 2>(); }

TEST(Storage, InnerRegionAlignmentChar021) { run<char, 0, 2, 1>(); }

TEST(Storage, InnerRegionAlignmentInt021) { run<int, 0, 2, 1>(); }

TEST(Storage, InnerRegionAlignmentFloat021) { run<float, 0, 2, 1>(); }

TEST(Storage, InnerRegionAlignmentDouble021) { run<double, 0, 2, 1>(); }
