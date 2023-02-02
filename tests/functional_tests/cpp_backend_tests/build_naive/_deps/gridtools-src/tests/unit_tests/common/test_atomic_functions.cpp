/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/atomic_functions.hpp>

#include <cstdlib>

#include <gtest/gtest.h>

#include <verifier.hpp>

template <typename T>
void TestAtomicAdd() {
    constexpr int size = 360;
    T field[size];
    T sum = 0;
    T sumRef = 0;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        sumRef += field[cnt];
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_add(sum, field[cnt]);
    }
    EXPECT_TRUE(gridtools::expect_with_threshold(sumRef, sum));
}

template <typename T>
void TestAtomicSub() {
    constexpr int size = 360;
    T field[size];
    T sum = 0;
    T sumRef = 0;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        sumRef -= field[cnt];
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_sub(sum, field[cnt]);
    }
    EXPECT_TRUE(gridtools::expect_with_threshold(sumRef, sum));
}

template <typename T>
void TestAtomicMin() {
    constexpr int size = 360;
    T field[size];
    T min = 99999;
    T minRef = 99999;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        minRef = std::min(minRef, field[cnt]);
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_min(min, field[cnt]);
    }
    EXPECT_EQ(minRef, min);
}

template <typename T>
void TestAtomicMax() {
    constexpr int size = 360;
    T field[size];
    T max = 0;
    T maxRef = 0;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        maxRef = std::max(maxRef, field[cnt]);
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_max(max, field[cnt]);
    }
    EXPECT_EQ(maxRef, max);
}

TEST(AtomicFunctionsUnittest, add) {
    TestAtomicAdd<int>();
    TestAtomicAdd<double>();
    TestAtomicAdd<float>();
}

TEST(AtomicFunctionsUnittest, sub) {
    TestAtomicSub<int>();
    TestAtomicSub<double>();
    TestAtomicSub<float>();
}

TEST(AtomicFunctionsUnittest, min) {
    TestAtomicMin<int>();
    TestAtomicMin<double>();
    TestAtomicMin<float>();
}

TEST(AtomicFunctionsUnittest, max) {
    TestAtomicMax<int>();
    TestAtomicMax<double>();
    TestAtomicMax<float>();
}
