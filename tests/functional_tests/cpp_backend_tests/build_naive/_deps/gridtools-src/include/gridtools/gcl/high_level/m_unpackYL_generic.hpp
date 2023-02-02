/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "../../common/array.hpp"
#include "../../common/halo_descriptor.hpp"
#include "wrap_argument.hpp"

template <typename value_type>
__global__ void m_unpackYLKernel_generic(value_type *__restrict__ d_data,
    value_type **__restrict__ d_msgbufTab_r,
    const wrap_argument d_msgsize_r,
    const gridtools::array<gridtools::halo_descriptor, 3> halo,
    int const nx,
    int const nz,
    const int traslation_const,
    const int field_index) {

    // per block shared buffer for storing destination buffers
    __shared__ value_type *msgbuf[27];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // load msg buffer table into shmem. Only the first 9 threads
    // need to do this
    if (threadIdx.x < 27 && threadIdx.y == 0 && threadIdx.z == 0) {
        msgbuf[threadIdx.x] = d_msgbufTab_r[threadIdx.x];
    }

    // an expression used later quite a bit
    int aa = halo[0].minus() + halo[0].end() - halo[0].begin();

    int pas = halo[0].minus();
    int ba = 1;
    int aas = 0;
    int la = halo[0].end() - halo[0].begin() + 1;
    if (idx < halo[0].minus()) {
        pas = 0;
        ba = 0;
        la = halo[0].minus();
    }
    if (idx > aa) {
        ba = 2;
        la = halo[0].plus();
        aas = halo[0].end() - halo[0].begin() + 1;
    }

    int bb = 0;
    int lb = halo[1].minus();

    int bc = 1;
    // int lc = halo[2].end() - halo[2].begin() + 1;

    int b_ind = ba + 3 * bb + 9 * bc;

    int oa = idx - pas - aas;
    int ob = idy;
    int oc = idz;

    int isrc = oa + ob * la + oc * la * lb + d_msgsize_r[b_ind];

    __syncthreads();
    value_type x;
    // store the data in the correct message buffer
    if ((idx < nx) && (idz < nz)) {
        x = msgbuf[b_ind][isrc];
    }

    int tli = halo[0].total_length();
    int tlj = halo[1].total_length();
    int idst = idx + idy * tli + idz * tli * tlj + traslation_const;

    if ((idx < nx) && (idz < nz)) {
        // printf("YL %d %d %d (%d) -> %16.16e\n", idx, idy, idz, idst, x);
        d_data[idst] = x;
    }

    // at this point we need to be sure that threads 0 - 8 have loaded the
    // message buffer table.
}

template <typename array_t>
void m_unpackYL_generic(
    array_t const &fields, typename array_t::value_type::value_type **d_msgbufTab_r, int *d_msgsize_r) {
    const int niter = fields.size();

    // run the compression a few times, just to get a bit
    // more statistics
    for (int i = 0; i < niter; i++) {

        // threads per block. Should be at least one warp in x, could be wider in y
        const int ntx = 32;
        const int nty = 1;
        const int ntz = 8;
        dim3 threads(ntx, nty, ntz);

        // form the grid to cover the entire plane. Use 1 block per z-layer
        int nx = fields[i].halos[0].r_length(-1) + fields[i].halos[0].r_length(0) + fields[i].halos[0].r_length(1);
        int ny = fields[i].halos[1].r_length(-1);
        int nz = fields[i].halos[2].r_length(0);

        int nbx = (nx + ntx - 1) / ntx;
        int nby = (ny + nty - 1) / nty;
        int nbz = (nz + ntz - 1) / ntz;
        dim3 blocks(nbx, nby, nbz);

        if (nbx != 0 && nby != 0 && nbz != 0) {
            // the actual kernel launch
            m_unpackYLKernel_generic<<<blocks, threads>>>(fields[i].ptr,
                reinterpret_cast<typename array_t::value_type::value_type **>(d_msgbufTab_r),
                wrap_argument(d_msgsize_r + 27 * i),
                *(reinterpret_cast<const gridtools::array<gridtools::halo_descriptor, 3> *>(&fields[i])),
                nx,
                nz,
                (fields[i].halos[0].begin() - fields[i].halos[0].minus()) +
                    (fields[i].halos[1].begin() - fields[i].halos[1].minus()) * fields[i].halos[0].total_length()
                    //         + (fields[i].halos[1].begin())*fields[i].halos[0].total_length()
                    + (fields[i].halos[2].begin()) * fields[i].halos[0].total_length() *
                          fields[i].halos[1].total_length(),
                0);
            GT_CUDA_CHECK(cudaGetLastError());
        }
    }
}
