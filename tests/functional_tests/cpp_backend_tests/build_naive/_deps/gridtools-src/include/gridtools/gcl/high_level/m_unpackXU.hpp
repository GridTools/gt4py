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
__global__ void m_unpackXUKernel(value_type *__restrict__ d_data,
    value_type **__restrict__ d_msgbufTab_r,
    const int *d_msgsize_r,
    const gridtools::halo_descriptor *halo /*_g*/,
    int const ny,
    int const nz,
    const int traslation_const,
    const int field_index) {

    // per block shared buffer for storing destination buffers
    __shared__ value_type *msgbuf[27];
    //__shared__ gridtools::halo_descriptor halo[3];

    int idx = blockIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // load msg buffer table into shmem. Only the first 9 threads
    // need to do this
    if (threadIdx.x == 0 && threadIdx.y < 27 && threadIdx.z == 0) {
        msgbuf[threadIdx.y] = d_msgbufTab_r[threadIdx.y];
    }

    int ba = 2;
    int la = halo[0].plus();

    int bb = 1;
    int lb = halo[1].end() - halo[1].begin() + 1;

    int bc = 1;

    int b_ind = ba + 3 * bb + 9 * bc;

    int oa = idx;
    int ob = idy;
    int oc = idz;

    int isrc = oa + ob * la + oc * la * lb + field_index * d_msgsize_r[b_ind];

    // at this point we need to be sure that threads 0 - 8 have loaded the
    // message buffer table.
    __syncthreads();

    value_type x;
    // store the data in the correct message buffer
    if ((idy < ny) && (idz < nz)) {
        x = msgbuf[b_ind][isrc];
    }

    // load the data from the contiguous source buffer
    int tli = halo[0].total_length();
    int tlj = halo[1].total_length();
    int idst = idx + idy * tli + idz * tli * tlj + traslation_const;

    if ((idy < ny) && (idz < nz)) {
        d_data[idst] = x;
    }
}

template <typename array_t, typename value_type>
void m_unpackXU(array_t const &d_data_array,
    value_type **d_msgbufTab_r,
    int d_msgsize_r[27],
    const gridtools::halo_descriptor halo[3],
    const gridtools::halo_descriptor halo_d[3]) {
    // threads per block. Should be at least one warp in x, could be wider in y
    const int ntx = 1;
    const int nty = 32;
    const int ntz = 8;
    dim3 threads(ntx, nty, ntz);

    // form the grid to cover the entire plane. Use 1 block per z-layer
    int nx = halo[0].r_length(1);
    int ny = halo[1].r_length(0);
    int nz = halo[2].r_length(0);

    int nbx = (nx + ntx - 1) / ntx;
    int nby = (ny + nty - 1) / nty;
    int nbz = (nz + ntz - 1) / ntz;
    dim3 blocks(nbx, nby, nbz);

    if (nbx == 0 || nby == 0 || nbz == 0)
        return;

    const int niter = d_data_array.size();

    // run the compression a few times, just to get a bit
    // more statistics
    for (int i = 0; i < niter; i++) {

        // the actual kernel launch
        m_unpackXUKernel<<<blocks, threads>>>(d_data_array[i],
            d_msgbufTab_r,
            d_msgsize_r,
            halo_d,
            ny,
            nz,
            (halo[0].end() + 1) + (halo[1].begin()) * halo[0].total_length() +
                (halo[2].begin()) * halo[0].total_length() * halo[1].total_length(),
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
int call_kernel_XU_u(Blocks blocks,
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
    m_unpackXUKernel<<<blocks, threads, b>>>(d_data, d_msgbufTab, d_msgsize, halo_d, nx, ny, tranlation_const, i);
    GT_CUDA_CHECK(cudaGetLastError());
    return 0;
}

template <typename value_type, typename datas, unsigned int... Ids>
void m_unpackXU_variadic(value_type **d_msgbufTab_r,
    int d_msgsize_r[27],
    const gridtools::halo_descriptor halo[3],
    const gridtools::halo_descriptor halo_d[3],
    const datas &d_datas,
    std::integer_sequence<unsigned int, Ids...>) {
    // threads per block. Should be at least one warp in x, could be wider in y
    const int ntx = 1;
    const int nty = 32;
    const int ntz = 8;
    dim3 threads(ntx, nty, ntz);

    // form the grid to cover the entire plane. Use 1 block per z-layer
    int nx = halo[0].r_length(1);
    int ny = halo[1].r_length(0);
    int nz = halo[2].r_length(0);

    int nbx = (nx + ntx - 1) / ntx;
    int nby = (ny + nty - 1) / nty;
    int nbz = (nz + ntz - 1) / ntz;
    dim3 blocks(nbx, nby, nbz);

    if (nbx == 0 || nby == 0 || nbz == 0)
        return;

    const int niter = std::tuple_size<datas>::value;
    int nothing[niter] = {call_kernel_XU_u(blocks,
        threads,
        0,
        static_cast<value_type *>(std::get<Ids>(d_datas)),
        d_msgbufTab_r,
        d_msgsize_r,
        halo_d,
        ny,
        nz,
        (halo[0].end() + 1) + (halo[1].begin()) * halo[0].total_length() +
            (halo[2].begin()) * halo[0].total_length() * halo[1].total_length(),
        Ids)...};
}
