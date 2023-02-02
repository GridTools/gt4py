/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#define GCL_MACRO_IMPL(z, n, _)                                                                                  \
    {                                                                                                            \
        const int ntx = 32;                                                                                      \
        const int nty = 8;                                                                                       \
        const int ntz = 1;                                                                                       \
        dim3 threads(ntx, nty, ntz);                                                                             \
                                                                                                                 \
        int nx = field##n.halos[0].s_length(-1) + field##n.halos[0].s_length(0) + field##n.halos[0].s_length(1); \
        int ny = field##n.halos[1].s_length(-1) + field##n.halos[1].s_length(0) + field##n.halos[1].s_length(1); \
        int nz = field##n.halos[2].s_length(1);                                                                  \
                                                                                                                 \
        int nbx = (nx + ntx - 1) / ntx;                                                                          \
        int nby = (ny + nty - 1) / nty;                                                                          \
        int nbz = (nz + ntz - 1) / ntz;                                                                          \
        dim3 blocks(nbx, nby, nbz);                                                                              \
                                                                                                                 \
        if (nbx != 0 && nby != 0 && nbz != 0) {                                                                  \
            m_packZUKernel_generic<<<blocks, threads, 0, 0>>>(field##n.ptr,                                      \
                reinterpret_cast<typename FOTF_T##n::value_type **>(d_msgbufTab),                                \
                wrap_argument(d_msgsize + 27 * n),                                                               \
                *(reinterpret_cast<const ::gridtools::array<::gridtools::halo_descriptor, 3> *>(&field##n)),     \
                nx,                                                                                              \
                ny,                                                                                              \
                0);                                                                                              \
        }                                                                                                        \
    }

BOOST_PP_REPEAT(GCL_NOI, GCL_MACRO_IMPL, all)
#undef GCL_MACRO_IMPL
