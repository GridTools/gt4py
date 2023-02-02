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

#include <set>

#include <gridtools/common/hugepage_alloc.hpp>

namespace gridtools {
    namespace {

        TEST(hugepage_alloc, ilog2) {
            for (std::size_t i = 0; i < 10; ++i)
                EXPECT_EQ(hugepage_alloc_impl_::ilog2(1 << i), i);
        }

        TEST(hugepage_alloc, cache_line_size) {
            EXPECT_GE(hugepage_alloc_impl_::cache_line_size(), sizeof(hugepage_alloc_impl_::ptr_metadata));
#ifdef __x86_64__
            EXPECT_EQ(hugepage_alloc_impl_::cache_line_size(), 64);
#endif
        }

        TEST(hugepage_alloc, cache_sets) { EXPECT_GT(hugepage_alloc_impl_::cache_sets(), 0); }

        TEST(hugepage_alloc, hugepage_size) {
            EXPECT_GE(hugepage_alloc_impl_::hugepage_size(), hugepage_alloc_impl_::page_size());
        }

        TEST(hugepage_alloc, page_size) { EXPECT_GT(hugepage_alloc_impl_::page_size(), 0); }

        struct hugepage_alloc_fixture : ::testing::TestWithParam<std::string> {
            std::string backup_mode;
            void SetUp() {
                const char *backup_mode_v = std::getenv("GT_HUGEPAGE_MODE");
                backup_mode = backup_mode_v ? backup_mode_v : "";
                setenv("GT_HUGEPAGE_MODE", GetParam().c_str(), 1);
            }
            void TearDown() {
                if (backup_mode.empty())
                    unsetenv("GT_HUGEPAGE_MODE");
                else
                    setenv("GT_HUGEPAGE_MODE", backup_mode.c_str(), 1);
            }
        };

        TEST_P(hugepage_alloc_fixture, hugepage_mode_from_env) {
            auto value = hugepage_alloc_impl_::hugepage_mode_from_env();
            auto expected = hugepage_alloc_impl_::hugepage_mode::transparent;
            if (GetParam() == "disable")
                expected = hugepage_alloc_impl_::hugepage_mode::disabled;
            else if (GetParam() == "explicit")
                expected = hugepage_alloc_impl_::hugepage_mode::explicit_allocation;
            EXPECT_EQ(value, expected);
        }

        TEST_P(hugepage_alloc_fixture, alloc_free) {
            std::size_t n = 100;

            int *ptr = static_cast<int *>(hugepage_alloc(n * sizeof(int)));
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % hugepage_alloc_impl_::cache_line_size(), 0);

            for (std::size_t i = 0; i < n; ++i) {
                ptr[i] = 0;
                EXPECT_EQ(ptr[i], 0);
            }

            hugepage_free(ptr);
        }

        TEST_P(hugepage_alloc_fixture, offsets) {
            // test shifting of the allocated data: hugepage_alloc guarantees that consecutive allocations return
            // pointers with different last X bits to reduce number of cache set conflict misses
            std::size_t cache_sets = hugepage_alloc_impl_::cache_sets();
            std::size_t different_bits = hugepage_alloc_impl_::ilog2(hugepage_alloc_impl_::cache_line_size()) +
                                         hugepage_alloc_impl_::ilog2(cache_sets);
            std::uintptr_t mask = (1ULL << different_bits) - 1;
            std::set<std::uintptr_t> offsets;
            for (std::size_t i = 0; i < cache_sets; ++i) {
                double *ptr = static_cast<double *>(hugepage_alloc(sizeof(double)));
                offsets.insert(reinterpret_cast<std::uintptr_t>(ptr) & mask);
                hugepage_free(ptr);
            }
#ifndef __cray__ // Cray clang version 10.0.2 seems to do a wrong optimization
            EXPECT_EQ(offsets.size(), cache_sets);
#endif
        }

        INSTANTIATE_TEST_SUITE_P(hugepage_alloc, hugepage_alloc_fixture, ::testing::Values("disable", "transparent"));

    } // namespace
} // namespace gridtools
