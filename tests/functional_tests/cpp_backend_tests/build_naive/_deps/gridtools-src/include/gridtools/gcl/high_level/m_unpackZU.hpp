/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <utility>

#include "../../common/halo_descriptor.hpp"

template <typename value_type>
__global__ void m_unpackZUKernel(value_type *__restrict__ d_data,
    value_type **__restrict__ d_msgbufTab_r,
    const int *d_msgsize,
    const gridtools::halo_descriptor *halo /*_g*/,
    int const nx,
    int const ny,
    int const translation_const,
    int field_index) {

    // per block shared buffer for storing destination buffers
    __shared__ value_type *msgbuf[27];
    //__shared__ gridtools::halo_descriptor halo[3];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    // load msg buffer table into shmem. Only the first 9 threads
    // need to do this
    if (threadIdx.x < 27 && threadIdx.y == 0) {
        msgbuf[threadIdx.x] = d_msgbufTab_r[threadIdx.x];
    }

    int aa = halo[0].minus() + halo[0].end() - halo[0].begin();
    int ab = halo[1].minus() + halo[1].end() - halo[1].begin();

    int pas = halo[0].minus();
    if (idx < halo[0].minus())
        pas = 0;

    int pbs = halo[1].minus();
    if (idy < halo[1].minus())
        pbs = 0;

    int ba = 1;
    int aas = 0;
    int la = halo[0].end() - halo[0].begin() + 1;
    if (idx < halo[0].minus()) {
        ba = 0;
        la = halo[0].minus();
    }
    if (idx > aa) {
        ba = 2;
        la = halo[0].plus();
        aas = halo[0].end() - halo[0].begin() + 1;
    }

    int bb = 1;
    int abs = 0;
    int lb = halo[1].end() - halo[1].begin() + 1;
    if (idy < halo[1].minus()) {
        bb = 0;
        lb = halo[1].minus();
    }
    if (idy > ab) {
        bb = 2;
        lb = halo[1].plus();
        abs = halo[1].end() - halo[1].begin() + 1;
    }

    int bc = 2;
    int lc = halo[2].plus();

    int b_ind = ba + 3 * bb + 9 * bc;
    int oa = idx - pas - aas;
    int ob = idy - pbs - abs;
    int oc = idz;

    int isrc = oa + ob * la + oc * la * lb + field_index * d_msgsize[b_ind];

    __syncthreads();
    value_type x;
    // store the data in the correct message buffer
    if ((idx < nx) && (idy < ny)) {
        x = msgbuf[b_ind][isrc];
    }

    int tli = halo[0].total_length();
    int tlj = halo[1].total_length();

    int idst = idx + idy * tli + idz * tli * tlj + translation_const;
    if ((idx < nx) && (idy < ny)) {
        d_data[idst] = x;
    }
}

template <typename array_t, typename value_type>
void m_unpackZU(array_t const &d_data_array,
    value_type **d_msgbufTab_r,
    int d_msgsize_r[27],
    const gridtools::halo_descriptor halo[3],
    const gridtools::halo_descriptor halo_d[3]) {
    // threads per block. Should be at least one warp in x, could be wider in y
    const int ntx = 32;
    const int nty = 8;
    const int ntz = 1;
    dim3 threads(ntx, nty, ntz);

    // form the grid to cover the entire plane. Use 1 block per z-layer
    int nx = halo[0].r_length(-1) + halo[0].r_length(0) + halo[0].r_length(1);
    int ny = halo[1].r_length(-1) + halo[1].r_length(0) + halo[1].r_length(1);
    int nz = halo[2].r_length(1);

    int nbx = (nx + ntx - 1) / ntx;
    int nby = (ny + nty - 1) / nty;
    int nbz = (nz + ntz - 1) / ntz;
    dim3 blocks(nbx, nby, nbz); // assuming halo[2].minus==halo[2].plus

    if (nbx == 0 || nby == 0 || nbz == 0)
        return;

    const int niter = d_data_array.size();

    // run the compression a few times, just to get a bit
    // more statistics
    for (int i = 0; i < niter; i++) {
        // the actual kernel launch
        m_unpackZUKernel<<<blocks, threads>>>(d_data_array[i],
            d_msgbufTab_r,
            d_msgsize_r,
            halo_d,
            nx,
            ny,
            (halo[0].begin() - halo[0].minus()) + (halo[1].begin() - halo[1].minus()) * halo[0].total_length() +
                (halo[2].end() + 1) * halo[0].total_length() * halo[1].total_length(),
            i);
        GT_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename Blocks,
    typename Threads,
    typename Bytes,
    typename Pointer,
    typename MsgbufTab,
    typename Msgsize,
    typename Halo>
int call_kernel_ZU_u(Blocks blocks,
    Threads threads,
    Bytes b,
    Pointer d_data,
    MsgbufTab d_msgbufTab,
    Msgsize d_msgsize,
    Halo halo_d,
    int nx,
    int ny,
    int tranlation_const,
    int i) {
    m_unpackZUKernel<<<blocks, threads, b>>>(d_data, d_msgbufTab, d_msgsize, halo_d, nx, ny, tranlation_const, i);

    GT_CUDA_CHECK(cudaGetLastError());

    return 0;
}

template <typename value_type, typename datas, unsigned int... Ids>
void m_unpackZU_variadic(value_type **d_msgbufTab_r,
    int d_msgsize_r[27],
    const gridtools::halo_descriptor halo[3],
    const gridtools::halo_descriptor halo_d[3],
    const datas &d_datas,
    std::integer_sequence<unsigned int, Ids...>) {
    // threads per block. Should be at least one warp in x, could be wider in y
    const int ntx = 32;
    const int nty = 8;
    const int ntz = 1;
    dim3 threads(ntx, nty, ntz);

    // form the grid to cover the entire plane. Use 1 block per z-layer
    int nx = halo[0].r_length(-1) + halo[0].r_length(0) + halo[0].r_length(1);
    int ny = halo[1].r_length(-1) + halo[1].r_length(0) + halo[1].r_length(1);
    int nz = halo[2].r_length(1);

    int nbx = (nx + ntx - 1) / ntx;
    int nby = (ny + nty - 1) / nty;
    int nbz = (nz + ntz - 1) / ntz;
    dim3 blocks(nbx, nby, nbz); // assuming halo[2].minus==halo[2].plus

    if (nbx == 0 || nby == 0 || nbz == 0)
        return;

    const int niter = std::tuple_size<datas>::value;
    int nothing[niter] = {call_kernel_ZU_u(blocks,
        threads,
        0,
        static_cast<value_type *>(std::get<Ids>(d_datas)),
        d_msgbufTab_r,
        d_msgsize_r,
        halo_d,
        nx,
        ny,
        (halo[0].begin() - halo[0].minus()) + (halo[1].begin() - halo[1].minus()) * halo[0].total_length() +
            (halo[2].end() + 1) * halo[0].total_length() * halo[1].total_length(),
        Ids)...};
}
