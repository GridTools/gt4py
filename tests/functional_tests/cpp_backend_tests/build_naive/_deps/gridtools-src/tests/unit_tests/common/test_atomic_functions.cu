/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"
#include <cstdlib>
#include <gridtools/common/atomic_functions.hpp>
#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/defs.hpp>

template <typename T>
struct verifier {
    static void TestEQ(T val, T exp) {
        T err = std::fabs(val - exp) / std::fabs(val);
        ASSERT_TRUE(err < 1e-12);
    }
};

template <>
struct verifier<float> {
    static void TestEQ(float val, float exp) {
        double err = std::fabs(val - exp) / std::fabs(val);
        ASSERT_TRUE(err < 1e-6);
    }
};

template <>
struct verifier<int> {
    static void TestEQ(int val, int exp) { ASSERT_EQ(val, exp); }
};

template <typename T>
__global__ void atomic_add_kernel(T *pReduced, const T *field, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = j * gridDim.x * blockDim.x + i;
    gridtools::atomic_add(*pReduced, field[pos]);
}

template <typename T>
__global__ void atomic_sub_kernel(T *pReduced, const T *field, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = j * gridDim.x * blockDim.x + i;
    gridtools::atomic_sub(*pReduced, field[pos]);
}
template <typename T>
__global__ void atomic_min_kernel(T *pReduced, const T *field, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = j * gridDim.x * blockDim.x + i;
    gridtools::atomic_min(*pReduced, field[pos]);
}
template <typename T>
__global__ void atomic_max_kernel(T *pReduced, const T *field, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = j * gridDim.x * blockDim.x + i;
    gridtools::atomic_max(*pReduced, field[pos]);
}

template <typename T>
void test_atomic_add() {
    dim3 threadsPerBlock(4, 4);
    dim3 numberOfBlocks(4, 4);

    int size = threadsPerBlock.x * threadsPerBlock.y * numberOfBlocks.x * numberOfBlocks.y;
    T field[size];

    T sumRef = 0;
    T sum = 0;
    T *sumDevice;
    GT_CUDA_CHECK(cudaMalloc(&sumDevice, sizeof(T)));
    GT_CUDA_CHECK(cudaMemcpy(sumDevice, &sum, sizeof(T), cudaMemcpyHostToDevice));

    T *fieldDevice;
    GT_CUDA_CHECK(cudaMalloc(&fieldDevice, sizeof(T) * size));

    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        sumRef += field[cnt];
    }

    GT_CUDA_CHECK(cudaMemcpy(fieldDevice, &field[0], sizeof(T) * size, cudaMemcpyHostToDevice));

    atomic_add_kernel<<<numberOfBlocks, threadsPerBlock>>>(sumDevice, fieldDevice, size);

    GT_CUDA_CHECK(cudaMemcpy(&sum, sumDevice, sizeof(T), cudaMemcpyDeviceToHost));
    verifier<T>::TestEQ(sumRef, sum);
}

template <typename T>
void test_atomic_sub() {
    dim3 threadsPerBlock(4, 4);
    dim3 numberOfBlocks(4, 4);

    int size = threadsPerBlock.x * threadsPerBlock.y * numberOfBlocks.x * numberOfBlocks.y;
    T field[size];

    T sumRef = 0;
    T sum = 0;
    T *sumDevice;
    GT_CUDA_CHECK(cudaMalloc(&sumDevice, sizeof(T)));
    GT_CUDA_CHECK(cudaMemcpy(sumDevice, &sum, sizeof(T), cudaMemcpyHostToDevice));

    T *fieldDevice;
    GT_CUDA_CHECK(cudaMalloc(&fieldDevice, sizeof(T) * size));

    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        sumRef -= field[cnt];
    }

    GT_CUDA_CHECK(cudaMemcpy(fieldDevice, &field[0], sizeof(T) * size, cudaMemcpyHostToDevice));

    atomic_sub_kernel<<<numberOfBlocks, threadsPerBlock>>>(sumDevice, fieldDevice, size);

    GT_CUDA_CHECK(cudaMemcpy(&sum, sumDevice, sizeof(T), cudaMemcpyDeviceToHost));
    verifier<T>::TestEQ(sumRef, sum);
}

template <typename T>
void test_atomic_min() {
    dim3 threadsPerBlock(4, 4);
    dim3 numberOfBlocks(4, 4);

    int size = threadsPerBlock.x * threadsPerBlock.y * numberOfBlocks.x * numberOfBlocks.y;
    T field[size];

    T minRef = 99999;
    T min = 99999;
    T *minDevice;
    GT_CUDA_CHECK(cudaMalloc(&minDevice, sizeof(T)));
    GT_CUDA_CHECK(cudaMemcpy(minDevice, &min, sizeof(T), cudaMemcpyHostToDevice));

    T *fieldDevice;
    GT_CUDA_CHECK(cudaMalloc(&fieldDevice, sizeof(T) * size));

    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        minRef = std::min(minRef, field[cnt]);
    }

    GT_CUDA_CHECK(cudaMemcpy(fieldDevice, &field[0], sizeof(T) * size, cudaMemcpyHostToDevice));

    atomic_min_kernel<<<numberOfBlocks, threadsPerBlock>>>(minDevice, fieldDevice, size);

    GT_CUDA_CHECK(cudaMemcpy(&min, minDevice, sizeof(T), cudaMemcpyDeviceToHost));
    verifier<T>::TestEQ(minRef, min);
}
template <typename T>
void test_atomic_max() {
    dim3 threadsPerBlock(4, 4);
    dim3 numberOfBlocks(4, 4);

    int size = threadsPerBlock.x * threadsPerBlock.y * numberOfBlocks.x * numberOfBlocks.y;
    T field[size];

    T maxRef = -1;
    T max = -1;
    T *maxDevice;
    GT_CUDA_CHECK(cudaMalloc(&maxDevice, sizeof(T)));
    GT_CUDA_CHECK(cudaMemcpy(maxDevice, &max, sizeof(T), cudaMemcpyHostToDevice));

    T *fieldDevice;
    GT_CUDA_CHECK(cudaMalloc(&fieldDevice, sizeof(T) * size));

    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast<T>(std::rand() % 100 + (std::rand() % 100) * 0.005);
        maxRef = std::max(maxRef, field[cnt]);
    }

    GT_CUDA_CHECK(cudaMemcpy(fieldDevice, &field[0], sizeof(T) * size, cudaMemcpyHostToDevice));

    atomic_max_kernel<<<numberOfBlocks, threadsPerBlock>>>(maxDevice, fieldDevice, size);

    GT_CUDA_CHECK(cudaMemcpy(&max, maxDevice, sizeof(T), cudaMemcpyDeviceToHost));
    verifier<T>::TestEQ(maxRef, max);
}

TEST(AtomicFunctionsUnittest, atomic_add_int) { test_atomic_add<int>(); }

TEST(AtomicFunctionsUnittest, atomic_add_real) {
    test_atomic_add<double>();
    test_atomic_add<float>();
}

TEST(AtomicFunctionsUnittest, atomic_sub_int) { test_atomic_sub<int>(); }

TEST(AtomicFunctionsUnittest, atomic_sub_real) {
    test_atomic_sub<double>();
    test_atomic_sub<float>();
}

TEST(AtomicFunctionsUnittest, atomic_min_int) { test_atomic_min<int>(); }
TEST(AtomicFunctionsUnittest, atomic_min_real) {
    test_atomic_min<double>();
    test_atomic_min<float>();
}

TEST(AtomicFunctionsUnittest, atomic_max_int) { test_atomic_max<int>(); }
TEST(AtomicFunctionsUnittest, atomic_max_real) {
    test_atomic_max<double>();
    test_atomic_max<float>();
}
