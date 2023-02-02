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
__global__ void m_packZUKernel_generic(const value_type *__restrict__ d_data,
    value_type **__restrict__ d_msgbufTab,
    const wrap_argument d_msgsize,
    const gridtools::array<gridtools::halo_descriptor, 3> halo,
    int const nx,
    int const ny,
    int const field_index) {

    // per block shared buffer for storing destination buffers
    __shared__ value_type *msgbuf[27];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    // load msg buffer table into shmem. Only the first 9 threads
    // need to do this
    if (threadIdx.x < 27 && threadIdx.y == 0) {
        msgbuf[threadIdx.x] = d_msgbufTab[threadIdx.x];
    }

    // an expression used later quite a bit
    int aa = halo[0].plus() + halo[0].end() - halo[0].begin();
    int ab = halo[1].plus() + halo[1].end() - halo[1].begin();
    int ac = halo[2].plus() + halo[2].end() - halo[2].begin();

    // load the data from the contiguous source buffer
    value_type x;
    int pas = 0, pbs = 0;
    pas = halo[0].plus();
    if (idx < halo[0].plus())
        pas = 0;

    pbs = halo[1].plus();
    if (idy < halo[1].plus())
        pbs = 0;

    int mas = 0;
    if (idx > aa)
        mas = halo[0].minus();

    int mbs = 0;
    if (idy > ab)
        mbs = halo[1].minus();

    int mcs = halo[2].minus();

    int ia = idx + halo[0].begin() - pas - mas;
    int ib = idy + halo[1].begin() - pbs - mbs;
    int ic = idz + halo[2].end() - mcs + 1;
    int isrc = ia + ib * halo[0].total_length() + ic * halo[0].total_length() * halo[1].total_length();

    if ((idx < nx) && (idy < ny)) {
        x = d_data[isrc];
        //     printf("ZU %e\n", x);
    }

    int ba = 1;
    int aas = 0;
    int la = halo[0].end() - halo[0].begin() + 1;
    if (idx < halo[0].plus()) {
        ba = 0;
        la = halo[0].plus();
    }
    if (idx > aa) {
        ba = 2;
        la = halo[0].minus();
        aas = halo[0].end() - halo[0].begin() + 1;
    }

    int bb = 1;
    int abs = 0;
    int lb = halo[1].end() - halo[1].begin() + 1;
    if (idy < halo[1].plus()) {
        bb = 0;
        lb = halo[1].plus();
    }
    if (idy > ab) {
        bb = 2;
        lb = halo[1].minus();
        abs = halo[1].end() - halo[1].begin() + 1;
    }

    int bc = 2;
    int lc = halo[2].minus();

    int b_ind = ba + 3 * bb + 9 * bc;

    int oa = idx - pas - aas;
    int ob = idy - pbs - abs;
    int oc = idz;

    int idst = oa + ob * la + oc * la * lb + d_msgsize[b_ind];
    // at this point we need to be sure that threads 0 - 8 have loaded the
    // message buffer table.
    __syncthreads();

    // store the data in the correct message buffer
    if ((idx < nx) && (idy < ny)) {
        msgbuf[b_ind][idst] = x;
    }
}

template <typename array_t>
void m_packZU_generic(array_t const &fields, typename array_t::value_type::value_type **d_msgbufTab, int *d_msgsize) {
    // run the compression a few times, just to get a bit
    // more statistics
    const int niter = fields.size();
    for (int i = 0; i < niter; i++) {

        // threads per block. Should be at least one warp in x, could be wider in y
        const int ntx = 32;
        const int nty = 8;
        const int ntz = 1;
        dim3 threads(ntx, nty, ntz);

        // form the grid to cover the entire plane. Use 1 block per z-layer
        int nx = fields[i].halos[0].s_length(-1) + fields[i].halos[0].s_length(0) + fields[i].halos[0].s_length(1);
        int ny = fields[i].halos[1].s_length(-1) + fields[i].halos[1].s_length(0) + fields[i].halos[1].s_length(1);
        int nz = fields[i].halos[2].s_length(1);

        int nbx = (nx + ntx - 1) / ntx;
        int nby = (ny + nty - 1) / nty;
        int nbz = (nz + ntz - 1) / ntz;
        dim3 blocks(nbx, nby, nbz);

        if (nbx != 0 && nby != 0 && nbz != 0) {
            // the actual kernel launch
            m_packZUKernel_generic<<<blocks, threads>>>(fields[i].ptr,
                (d_msgbufTab),
                wrap_argument(d_msgsize + 27 * i),
                *(reinterpret_cast<const gridtools::array<gridtools::halo_descriptor, 3> *>(&fields[i])),
                nx,
                ny,
                0);
            GT_CUDA_CHECK(cudaGetLastError());
        }
    }
}
