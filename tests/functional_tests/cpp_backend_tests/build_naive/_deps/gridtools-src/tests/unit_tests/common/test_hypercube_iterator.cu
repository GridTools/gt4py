/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "./test_hypercube_iterator.cpp"
#include <gridtools/common/cuda_util.hpp>

static const size_t Size = 2;

GT_FUNCTION int linear_index(gridtools::array<size_t, 2> &index) { return index[0] * Size + index[1]; }

__global__ void test_kernel(int *out_ptr) {
    for (size_t i = 0; i < Size * Size; ++i)
        out_ptr[i] = -1;

    using hypercube_t = gridtools::array<gridtools::array<size_t, 2>, 2>;
    for (auto pos : make_hypercube_view(hypercube_t{{{0ul, Size}, {0ul, Size}}})) {
        out_ptr[linear_index(pos)] = linear_index(pos);
    }
};

TEST(multi_iterator, iterate_on_device) {
    int *out;
    GT_CUDA_CHECK(cudaMalloc(&out, sizeof(int) * Size * Size));

    test_kernel<<<1, 1>>>(out);

    int host_out[Size * Size];
    GT_CUDA_CHECK(cudaMemcpy(&host_out, out, sizeof(int) * Size * Size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < Size * Size; ++i)
        ASSERT_EQ(i, host_out[i]) << "at i = " << i;
}
