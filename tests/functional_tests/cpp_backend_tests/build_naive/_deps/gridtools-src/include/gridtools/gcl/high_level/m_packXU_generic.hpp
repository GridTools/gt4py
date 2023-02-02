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
__global__ void m_packXUKernel_generic(const value_type *__restrict__ d_data,
    value_type **__restrict__ d_msgbufTab,
    const wrap_argument d_msgsize,
    const gridtools::array<gridtools::halo_descriptor, 3> halo,
    int const ny,
    int const nz,
    int const field_index) {

    // per block shared buffer for storing destination buffers
    __shared__ value_type *msgbuf[27];

    int idx = blockIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // load msg buffer table into shmem. Only the first 9 threads
    // need to do this
    if (threadIdx.x == 0 && threadIdx.y < 27 && threadIdx.z == 0) {
        msgbuf[threadIdx.y] = d_msgbufTab[threadIdx.y];
    }

    // load the data from the contiguous source buffer
    value_type x;
    if ((idy < ny) && (idz < nz)) {
        int mas = halo[0].minus();

        int ia = idx + halo[0].end() - mas + 1;
        ;
        int ib = idy + halo[1].begin();
        int ic = idz + halo[2].begin();
        int isrc = ia + ib * halo[0].total_length() + ic * halo[0].total_length() * halo[1].total_length();
        x = d_data[isrc];
        //     printf("XU %e\n", x);
    }

    int ba = 2;
    int la = halo[0].minus();

    int bb = 1;
    int lb = halo[1].end() - halo[1].begin() + 1;

    int bc = 1;

    int b_ind = ba + 3 * bb + 9 * bc;

    int oa = idx;
    int ob = idy;
    int oc = idz;

    int idst = oa + ob * la + oc * la * lb + d_msgsize[b_ind];
    ;

    // at this point we need to be sure that threads 0 - 8 have loaded the
    // message buffer table.
    __syncthreads();

    // store the data in the correct message buffer
    if ((idy < ny) && (idz < nz)) {
        // printf("XU %d %d %d -> %16.16e\n", idx, idy, idz, x);
        msgbuf[b_ind][idst] = x;
    }
}

template <typename array_t>
void m_packXU_generic(array_t const &fields, typename array_t::value_type::value_type **d_msgbufTab, int *d_msgsize) {
    const int niter = fields.size();

    // run the compression a few times, just to get a bit
    // more statistics
    for (int i = 0; i < niter; i++) {

        // threads per block. Should be at least one warp in x, could be wider in y
        const int ntx = 1;
        const int nty = 32;
        const int ntz = 8;
        dim3 threads(ntx, nty, ntz);

        // form the grid to cover the entire plane. Use 1 block per z-layer
        int nx = fields[i].halos[0].s_length(1);
        int ny = fields[i].halos[1].s_length(0);
        int nz = fields[i].halos[2].s_length(0);

        int nbx = (nx + ntx - 1) / ntx;
        int nby = (ny + nty - 1) / nty;
        int nbz = (nz + ntz - 1) / ntz;
        dim3 blocks(nbx, nby, nbz);

        if (nbx != 0 && nby != 0 && nbz != 0) {
            // the actual kernel launch
            m_packXUKernel_generic<<<blocks, threads>>>(fields[i].ptr,
                reinterpret_cast<typename array_t::value_type::value_type **>(d_msgbufTab),
                wrap_argument(d_msgsize + 27 * i),
                *(reinterpret_cast<const gridtools::array<gridtools::halo_descriptor, 3> *>(&fields[i])),
                ny,
                nz,
                0);
            GT_CUDA_CHECK(cudaGetLastError());
        }
    }
}
