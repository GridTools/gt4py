/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/gcl/all_to_all_halo.hpp>

#include <vector>

#include <mpi.h>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/halo_descriptor.hpp>
#include <gridtools/gcl/GCL.hpp>
#include <gridtools/gcl/low_level/proc_grids_3D.hpp>

using namespace gridtools;
using namespace gcl;

TEST(gcl, test_all_to_all_halo_3D) {
    constexpr int N = 13;
    constexpr int H = 6;

    typedef array<halo_descriptor, 3> halo_block;

    typedef MPI_3D_process_grid_t<3> grid_type;

    array<int, 3> dims{0, 0, 0};
    grid_type pgrid({true, true, true}, MPI_COMM_WORLD, dims);

    all_to_all_halo<int, grid_type> a2a(pgrid);

    int pi, pj, pk;
    int PI, PJ, PK;
    pgrid.coords(pi, pj, pk);
    pgrid.dims(PI, PJ, PK);

    std::vector<int> dataout(PI * N * PJ * N * PK * N);
    std::vector<int> datain((N + 2 * H) * (N + 2 * H) * (N + 2 * H));

    array<int, 3> crds;

    if (pid() == 0) {
        halo_block send_block;

        for (int i = 0; i < PI; ++i) {
            for (int j = 0; j < PJ; ++j) {
                for (int k = 0; k < PK; ++k) {

                    crds[0] = i;
                    crds[1] = j; // DECREASING STRIDES
                    crds[2] = k; // DECREASING STRIDES

                    // DECREASING STRIDES
                    send_block[0] = halo_descriptor(0, 0, i * N, (i + 1) * N - 1, PI * N);
                    send_block[1] = halo_descriptor(0, 0, j * N, (j + 1) * N - 1, PJ * N);
                    send_block[2] = halo_descriptor(0, 0, k * N, (k + 1) * N - 1, N * PK);

                    a2a.register_block_to(&dataout[0], send_block, crds);
                }
            }
        }
    }

    crds[0] = 0;
    crds[1] = 0; // DECREASING STRIDES
    crds[2] = 0; // DECREASING STRIDES

    // INCREASING STRIDES
    halo_block recv_block;
    recv_block[0] = halo_descriptor(H, H, H, N + H - 1, N + 2 * H);
    recv_block[1] = halo_descriptor(H, H, H, N + H - 1, N + 2 * H);
    recv_block[2] = halo_descriptor(H, H, H, N + H - 1, N + 2 * H);

    a2a.register_block_from(&datain[0], recv_block, crds);

    for (int i = 0; i < PI * N; ++i)
        for (int j = 0; j < PJ * N; ++j)
            for (int k = 0; k < PK * N; ++k)
                dataout[i * (PJ * N) * (PK * N) + j * (PK * N) + k] = i * (PJ * N) * (PK * N) + j * (PK * N) + k;

    for (int i = 0; i < N + 2 * H; ++i)
        for (int j = 0; j < N + 2 * H; ++j)
            for (int k = 0; k < N + 2 * H; ++k)
                datain[i * (N + 2 * H) * (N + 2 * H) + j * (N + 2 * H) + k] = 0;

    a2a.setup();
    a2a.start_exchange();
    a2a.wait();

    int stride0 = (N * PK * N * PJ);
    int stride1 = (N * PK);
    int offseti = pi * N;
    int offsetj = pj * N;
    int offsetk = pk * N;

    for (int i = H; i < N + H; ++i)
        for (int j = H; j < N + H; ++j)
            for (int k = H; k < N + H; ++k)
                EXPECT_EQ(dataout[(offseti + i - H) * stride0 + (offsetj + j - H) * stride1 + offsetk + k - H],
                    datain[i * (N + 2 * H) * (N + 2 * H) + j * (N + 2 * H) + k]);
}
