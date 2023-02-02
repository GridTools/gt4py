/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../low_level/arch.hpp"
#include "../low_level/translate.hpp"

#include "descriptors_fwd.hpp"

#ifdef GT_CUDACC
#include "../../common/cuda_runtime.hpp"
#include "../../common/cuda_util.hpp"
#endif

namespace gridtools {
    namespace gcl {
        template <class Layout, int I>
        int nth(int i, int j, int k) {
            int arr[] = {i, j, k};
            return arr[Layout::at(I)];
        }

        template <typename T, typename arch>
        struct gcl_alloc;

        template <typename T>
        struct gcl_alloc<T, cpu> {

            static T *alloc(size_t sz) {
                if (sz)
                    return new T[sz];
                else
                    return nullptr;
            }

            static void free(T *t) { delete[] t; }
        };

#ifdef GT_CUDACC
        template <typename T>
        struct gcl_alloc<T, gpu> {

            static T *alloc(size_t sz) {
                if (sz) {
                    T *ptr;
                    GT_CUDA_CHECK(cudaMalloc(&ptr, sz * sizeof(T)));
                    return ptr;
                } else {
                    return nullptr;
                }
            }

            static void free(T *t) {
                /* TODO: fix memory leak and related issues (just uncommenting breaks GCL in some rare cases) */
                // GT_CUDA_CHECK(cudaFree(t));
            }
        };
#endif

        template <typename T>
        struct allocation_service;

        template <typename Datatype, typename T2>
        struct allocation_service<hndlr_descriptor_ut<Datatype, T2>> {
            void operator()(hndlr_descriptor_ut<Datatype, T2> *hm) const {
                typedef typename hndlr_descriptor_ut<Datatype, T2>::pattern_type::translate_type translate;
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                hm->send_buffer[translate()(ii, jj, kk)] =
                                    gcl_alloc<Datatype, cpu>::alloc(hm->total_pack_size({ii, jj, kk}));
                                hm->recv_buffer[translate()(ii, jj, kk)] =
                                    gcl_alloc<Datatype, cpu>::alloc(hm->total_unpack_size({ii, jj, kk}));

                                hm->m_haloexch.register_send_to_buffer(&(hm->send_buffer[translate()(ii, jj, kk)][0]),
                                    hm->total_pack_size({ii, jj, kk}) * sizeof(Datatype),
                                    ii,
                                    jj,
                                    kk);

                                hm->m_haloexch.register_receive_from_buffer(
                                    &(hm->recv_buffer[translate()(ii, jj, kk)][0]),
                                    hm->total_unpack_size({ii, jj, kk}) * sizeof(Datatype),
                                    ii,
                                    jj,
                                    kk);
                            }
            }
        };

        template <typename Datatype, typename T2, typename procmap, typename arch, template <int Ndim> class GridType>
        struct allocation_service<hndlr_dynamic_ut<Datatype, GridType<3>, T2, procmap, arch>> {
            void operator()(hndlr_dynamic_ut<Datatype, GridType<3>, T2, procmap, arch> *hm, int mf) const {
                typedef translate_t<3> translate;
                typedef translate_t<3, procmap> translate_P;

                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                hm->send_size[translate()(ii, jj, kk)] = hm->halo.send_buffer_size({ii, jj, kk});
                                hm->recv_size[translate()(ii, jj, kk)] = hm->halo.recv_buffer_size({ii, jj, kk});
                                hm->send_buffer[translate()(ii, jj, kk)] =
                                    gcl_alloc<Datatype, arch>::alloc(hm->halo.send_buffer_size({ii, jj, kk}) * mf);
                                hm->recv_buffer[translate()(ii, jj, kk)] =
                                    gcl_alloc<Datatype, arch>::alloc(hm->halo.recv_buffer_size({ii, jj, kk}) * mf);

                                typedef typename translate_P::map_type map_type;
                                const int ii_P = nth<map_type, 0>(ii, jj, kk);
                                const int jj_P = nth<map_type, 1>(ii, jj, kk);
                                const int kk_P = nth<map_type, 2>(ii, jj, kk);

                                hm->m_haloexch.register_send_to_buffer(&(hm->send_buffer[translate()(ii, jj, kk)][0]),
                                    hm->halo.send_buffer_size({ii, jj, kk}) * sizeof(Datatype) * mf,
                                    ii_P,
                                    jj_P,
                                    kk_P);

                                hm->m_haloexch.register_receive_from_buffer(
                                    &(hm->recv_buffer[translate()(ii, jj, kk)][0]),
                                    hm->halo.recv_buffer_size({ii, jj, kk}) * sizeof(Datatype) * mf,
                                    ii_P,
                                    jj_P,
                                    kk_P);
                            }
            }
        };

        template <typename T>
        struct pack_service;

        template <typename Datatype, typename T2>
        struct pack_service<hndlr_descriptor_ut<Datatype, T2>> {
            void operator()(hndlr_descriptor_ut<Datatype, T2> const *hm) const {
                typedef typename hndlr_descriptor_ut<Datatype, T2>::pattern_type::translate_type translate;
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if ((ii != 0 || jj != 0 || kk != 0) && (hm->pattern().proc_grid().proc(ii, jj, kk) != -1)) {
                                Datatype *it = &(hm->send_buffer[translate()(ii, jj, kk)][0]);
                                for (int df = 0; df < hm->size(); ++df)
                                    hm->data_field(df).pack({ii, jj, kk}, it);
                            }
            }
        };

        template <typename T>
        struct unpack_service;

        template <typename Datatype, typename T2>
        struct unpack_service<hndlr_descriptor_ut<Datatype, T2>> {
            void operator()(hndlr_descriptor_ut<Datatype, T2> const *hm) const {
                typedef typename hndlr_descriptor_ut<Datatype, T2>::pattern_type::translate_type translate;
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if ((ii != 0 || jj != 0 || kk != 0) && (hm->pattern().proc_grid().proc(ii, jj, kk) != -1)) {
                                Datatype *it = &(hm->recv_buffer[translate()(ii, jj, kk)][0]);
                                for (int df = 0; df < hm->size(); ++df)
                                    hm->data_field(df).unpack({ii, jj, kk}, it);
                            }
            }
        };
    } // namespace gcl
} // namespace gridtools
