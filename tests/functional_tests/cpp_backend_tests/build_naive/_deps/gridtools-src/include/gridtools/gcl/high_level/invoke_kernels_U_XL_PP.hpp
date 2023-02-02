/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#define GCL_MACRO_IMPL(z, n, _)                                                                                        \
    {                                                                                                                  \
        const int ntx = 1;                                                                                             \
        const int nty = 32;                                                                                            \
        const int ntz = 8;                                                                                             \
        dim3 threads(ntx, nty, ntz);                                                                                   \
                                                                                                                       \
        int nx = field##n.halos[0].r_length(-1);                                                                       \
        int ny = field##n.halos[1].r_length(0);                                                                        \
        int nz = field##n.halos[2].r_length(0);                                                                        \
                                                                                                                       \
        int nbx = (nx + ntx - 1) / ntx;                                                                                \
        int nby = (ny + nty - 1) / nty;                                                                                \
        int nbz = (nz + ntz - 1) / ntz;                                                                                \
        dim3 blocks(nbx, nby, nbz);                                                                                    \
                                                                                                                       \
        if (nbx != 0 && nby != 0 && nbz != 0) {                                                                        \
            m_unpackXLKernel_generic<<<blocks, threads>>>(field##n.ptr,                                                \
                reinterpret_cast<typename FOTF_T##n::value_type **>(d_msgbufTab_r),                                    \
                wrap_argument(d_msgsize_r + 27 * n),                                                                   \
                *(reinterpret_cast<const ::gridtools::array<::gridtools::halo_descriptor, 3> *>(&field##n)),           \
                ny,                                                                                                    \
                nz,                                                                                                    \
                (field##n.halos[0].begin() - field##n.halos[0].minus()) +                                              \
                    (field##n.halos[1].begin()) * field##n.halos[0].total_length() +                                   \
                    (field##n.halos[2].begin()) * field##n.halos[0].total_length() * field##n.halos[1].total_length(), \
                0);                                                                                                    \
        }                                                                                                              \
    }

BOOST_PP_REPEAT(GCL_NOI, GCL_MACRO_IMPL, all)
#undef GCL_MACRO_IMPL
