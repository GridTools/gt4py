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

#include "../../common/defs.hpp"
#include "descriptor_base.hpp"

#ifdef GT_CUDACC
#include "m_packXL_generic.hpp"
#include "m_packXU_generic.hpp"
#include "m_packYL_generic.hpp"
#include "m_packYU_generic.hpp"
#include "m_packZL_generic.hpp"
#include "m_packZU_generic.hpp"

#include "m_unpackXL_generic.hpp"
#include "m_unpackXU_generic.hpp"
#include "m_unpackYL_generic.hpp"
#include "m_unpackYU_generic.hpp"
#include "m_unpackZL_generic.hpp"
#include "m_unpackZU_generic.hpp"

#define GCL_KERNEL_TYPE ZL
#include "call_generic.hpp"
#undef GCL_KERNEL_TYPE

#define GCL_KERNEL_TYPE ZU
#include "call_generic.hpp"
#undef GCL_KERNEL_TYPE

#define GCL_KERNEL_TYPE YL
#include "call_generic.hpp"
#undef GCL_KERNEL_TYPE

#define GCL_KERNEL_TYPE YU
#include "call_generic.hpp"
#undef GCL_KERNEL_TYPE

#define GCL_KERNEL_TYPE XL
#include "call_generic.hpp"
#undef GCL_KERNEL_TYPE

#define GCL_KERNEL_TYPE XU
#include "call_generic.hpp"
#undef GCL_KERNEL_TYPE
#endif

#include <vector>

#include "../../common/array.hpp"
#include "../low_level/translate.hpp"
#include "field_on_the_fly.hpp"
#include "helpers_impl.hpp"
#include "numerics.hpp"

namespace gridtools {
    namespace gcl {
        template <typename HaloExch, typename proc_layout_abs>
        class hndlr_generic<HaloExch, proc_layout_abs, cpu> : public descriptor_base<HaloExch> {
            using DIMS_t = std::integral_constant<int, 3>;
            array<char *, static_pow3(DIMS_t::value)> send_buffer; // One entry will not be used...
            array<char *, static_pow3(DIMS_t::value)> recv_buffer;
            array<int, static_pow3(DIMS_t::value)> send_buffer_size; // One entry will not be used...
            array<int, static_pow3(DIMS_t::value)> recv_buffer_size;

          public:
            typedef descriptor_base<HaloExch> base_type;
            typedef typename base_type::pattern_type pattern_type;
            /** Architecture type
             */
            typedef cpu arch_type;

            /**
               Type of the computin grid associated to the pattern
             */
            typedef typename base_type::grid_type grid_type;

            /**
               Type of the translation used to map dimensions to buffer addresses
             */
            typedef translate_t<DIMS_t::value> translate;

            hndlr_generic(grid_type const &g)
                : base_type(g), send_buffer{nullptr}, recv_buffer{nullptr}, send_buffer_size{0}, recv_buffer_size{0} {}

            ~hndlr_generic() {
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        for (int k = -1; k <= 1; ++k) {
                            gcl_alloc<char, arch_type>::free(send_buffer[translate()(i, j, k)]);
                            gcl_alloc<char, arch_type>::free(recv_buffer[translate()(i, j, k)]);
                        }
            }

            /**
               Setup function, in this version, takes tree parameters to
               compute internal buffers and sizes. It takes a field on the fly
               struct, which requires Datatype and layout map template
               arguments that are inferred, so the user is not aware of them.

               \tparam DataType This type is inferred by halo_example paramter
               \tparam t_layoutmap This type is inferred by halo_example paramter

               \param[in] max_fields_n Maximum number of grids used in a computation
               \param[in] halo_example The (at least) maximal grid that is goinf to be used
               \param[in] typesize In case the DataType of the halo_example is not the same as the maximum data type
               used in the computation, this parameter can be given
             */
            template <typename DataType, typename f_layoutmap, template <typename> class traits>
            void setup(int max_fields_n,
                field_on_the_fly<DataType, f_layoutmap, traits> const &halo_example,
                int typesize = sizeof(DataType)) {

                typedef typename field_on_the_fly<DataType, f_layoutmap, traits>::inner_layoutmap t_layoutmap;
                array<int, DIMS_t::value> eta;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        for (int k = -1; k <= 1; ++k) {
                            if (i != 0 || j != 0 || k != 0) {
                                eta[0] = i;
                                eta[1] = j;
                                eta[2] = k;
                                int S = 1;
                                S = halo_example.send_buffer_size(eta);
                                int R = 1;
                                R = halo_example.recv_buffer_size(eta);

                                send_buffer_size[translate()(i, j, k)] = (S * max_fields_n * typesize);
                                recv_buffer_size[translate()(i, j, k)] = (R * max_fields_n * typesize);

                                send_buffer[translate()(i, j, k)] =
                                    gcl_alloc<char, arch_type>::alloc(send_buffer_size[translate()(i, j, k)]);
                                recv_buffer[translate()(i, j, k)] =
                                    gcl_alloc<char, arch_type>::alloc(recv_buffer_size[translate()(i, j, k)]);

                                using proc_layout = layout_transform<t_layoutmap, proc_layout_abs>;
                                const int i_P = nth<proc_layout, 0>(i, j, k);
                                const int j_P = nth<proc_layout, 1>(i, j, k);
                                const int k_P = nth<proc_layout, 2>(i, j, k);

                                base_type::m_haloexch.register_send_to_buffer(&(send_buffer[translate()(i, j, k)][0]),
                                    send_buffer_size[translate()(i, j, k)],
                                    i_P,
                                    j_P,
                                    k_P);

                                base_type::m_haloexch.register_receive_from_buffer(
                                    &(recv_buffer[translate()(i, j, k)][0]),
                                    recv_buffer_size[translate()(i, j, k)],
                                    i_P,
                                    j_P,
                                    k_P);
                            }
                        }
                    }
                }
            }

            /**
               Setup function, in this version, takes a single parameter with
               an array of sizes to be associated with the halos.

               \tparam DataType This type is inferred by halo_example paramter
               \tparam t_layoutmap This type is inferred by halo_example paramter

               \param[in] buffer_size_list Array (gridtools::array) with the sizes of the buffers associated with the
               halos.
             */
            template <typename DataType, typename t_layoutmap>
            void setup(array<size_t, static_pow3(DIMS_t::value)> const &buffer_size_list) {
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        for (int k = -1; k <= 1; ++k) {
                            if (i != 0 || j != 0 || k != 0) {
                                send_buffer[translate()(i, j, k)] =
                                    gcl_alloc<char, arch_type>::alloc(buffer_size_list[translate()(i, j, k)]);
                                recv_buffer[translate()(i, j, k)] =
                                    gcl_alloc<char, arch_type>::alloc(buffer_size_list[translate()(i, j, k)]);
                                send_buffer_size[translate()(i, j, k)] = (buffer_size_list[translate()(i, j, k)]);
                                recv_buffer_size[translate()(i, j, k)] = (buffer_size_list[translate()(i, j, k)]);

                                using proc_layout = layout_transform<t_layoutmap, proc_layout_abs>;
                                const int i_P = nth<proc_layout, 0>(i, j, k);
                                const int j_P = nth<proc_layout, 1>(i, j, k);
                                const int k_P = nth<proc_layout, 2>(i, j, k);

                                base_type::m_haloexch.register_send_to_buffer(&(send_buffer[translate()(i, j, k)][0]),
                                    buffer_size_list[translate()(i, j, k)],
                                    i_P,
                                    j_P,
                                    k_P);

                                base_type::m_haloexch.register_receive_from_buffer(
                                    &(recv_buffer[translate()(i, j, k)][0]),
                                    buffer_size_list[translate()(i, j, k)],
                                    i_P,
                                    j_P,
                                    k_P);
                            }
                        }
                    }
                }
            }

            template <typename... FIELDS>
            void pack(const FIELDS &..._fields) const {
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            char *it = reinterpret_cast<char *>(&(send_buffer[translate()(ii, jj, kk)][0]));
                            pack_dims<DIMS_t::value, 0>()(*this, ii, jj, kk, it, _fields...);
                        }
                    }
                }
            }

            template <typename... FIELDS>
            void unpack(const FIELDS &..._fields) const {
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            char *it = reinterpret_cast<char *>(&(recv_buffer[translate()(ii, jj, kk)][0]));
                            unpack_dims<DIMS_t::value, 0>()(*this, ii, jj, kk, it, _fields...);
                        }
                    }
                }
            }

            /**
               Function to unpack received data

               \tparam array_of_fotf this should be an array of field_on_the_fly
               \param[in] fields vector with fields on the fly
            */
            template <typename T1, typename T2, template <typename> class T3>
            void pack(std::vector<field_on_the_fly<T1, T2, T3>> const &fields) {
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typename field_on_the_fly<T1, T2, T3>::value_type *it =
                                reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type *>(
                                    &(send_buffer[translate()(ii, jj, kk)][0]));
                            pack_vector_dims<DIMS_t::value, 0>()(*this, ii, jj, kk, it, fields);
                        }
                    }
                }
            }

            /**
               Function to unpack received data

               \tparam array_of_fotf this should be an array of field_on_the_fly
               \param[in] fields vector with fields on the fly
            */
            template <typename T1, typename T2, template <typename> class T3>
            void unpack(std::vector<field_on_the_fly<T1, T2, T3>> const &fields) {
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typename field_on_the_fly<T1, T2, T3>::value_type *it =
                                reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type *>(
                                    &(recv_buffer[translate()(ii, jj, kk)][0]));
                            unpack_vector_dims<DIMS_t::value, 0>()(*this, ii, jj, kk, it, fields);
                        }
                    }
                }
            }

          private:
            template <int, int>
            struct pack_dims {};

            template <int dummy>
            struct pack_dims<3, dummy> {

                template <typename T, typename iterator>
                void operator()(const T &, int, int, int, iterator &) const {}

                template <typename T, typename iterator, typename FIRST, typename... FIELDS>
                void operator()(
                    const T &hm, int ii, int jj, int kk, iterator &it, FIRST const &first, const FIELDS &..._fields)
                    const {
                    using proc_layout = layout_transform<typename FIRST::inner_layoutmap, proc_layout_abs>;
                    const int ii_P = nth<proc_layout, 0>(ii, jj, kk);
                    const int jj_P = nth<proc_layout, 1>(ii, jj, kk);
                    const int kk_P = nth<proc_layout, 2>(ii, jj, kk);
                    if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                        first.pack({ii, jj, kk}, first.ptr, it);
                        operator()(hm, ii, jj, kk, it, _fields...);
                    }
                }
            };

            template <int, int>
            struct unpack_dims {};

            template <int dummy>
            struct unpack_dims<3, dummy> {

                template <typename T, typename iterator>
                void operator()(const T &, int, int, int, iterator &) const {}

                template <typename T, typename iterator, typename FIRST, typename... FIELDS>
                void operator()(
                    const T &hm, int ii, int jj, int kk, iterator &it, FIRST const &first, const FIELDS &..._fields)
                    const {
                    using proc_layout = layout_transform<typename FIRST::inner_layoutmap, proc_layout_abs>;
                    const int ii_P = nth<proc_layout, 0>(ii, jj, kk);
                    const int jj_P = nth<proc_layout, 1>(ii, jj, kk);
                    const int kk_P = nth<proc_layout, 2>(ii, jj, kk);
                    if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                        first.unpack({ii, jj, kk}, first.ptr, it);
                        operator()(hm, ii, jj, kk, it, _fields...);
                    }
                }
            };

            template <int, int>
            struct pack_vector_dims {};

            template <int dummy>
            struct pack_vector_dims<3, dummy> {

                template <typename T, typename iterator, typename array_of_fotf>
                void operator()(const T &hm, int ii, int jj, int kk, iterator &it, array_of_fotf const &_fields) const {
                    using proc_layout =
                        layout_transform<typename array_of_fotf::value_type::inner_layoutmap, proc_layout_abs>;
                    const int ii_P = nth<proc_layout, 0>(ii, jj, kk);
                    const int jj_P = nth<proc_layout, 1>(ii, jj, kk);
                    const int kk_P = nth<proc_layout, 2>(ii, jj, kk);
                    if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                        for (unsigned int fi = 0; fi < _fields.size(); ++fi) {
                            _fields[fi].pack({ii, jj, kk}, _fields[fi].ptr, it);
                        }
                    }
                }
            };

            template <int, int>
            struct unpack_vector_dims {};

            template <int dummy>
            struct unpack_vector_dims<3, dummy> {

                template <typename T, typename iterator, typename array_of_fotf>
                void operator()(const T &hm, int ii, int jj, int kk, iterator &it, array_of_fotf const &_fields) const {
                    using proc_layout =
                        layout_transform<typename array_of_fotf::value_type::inner_layoutmap, proc_layout_abs>;
                    const int ii_P = nth<proc_layout, 0>(ii, jj, kk);
                    const int jj_P = nth<proc_layout, 1>(ii, jj, kk);
                    const int kk_P = nth<proc_layout, 2>(ii, jj, kk);
                    if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                        for (unsigned int fi = 0; fi < _fields.size(); ++fi) {
                            _fields[fi].unpack({ii, jj, kk}, _fields[fi].ptr, it);
                        }
                    }
                }
            };
        };

#ifdef GT_CUDACC
        template <typename HaloExch, typename proc_layout_abs>
        class hndlr_generic<HaloExch, proc_layout_abs, gpu> : public descriptor_base<HaloExch> {
            typedef gpu arch_type;

            using DIMS_t = std::integral_constant<int, 3>;
            array<char *, static_pow3(DIMS_t::value)> send_buffer; // One entry will not be used...
            array<char *, static_pow3(DIMS_t::value)> recv_buffer;
            array<int, static_pow3(DIMS_t::value)> send_buffer_size; // One entry will not be used...
            array<int, static_pow3(DIMS_t::value)> recv_buffer_size;
            char **d_send_buffer;
            char **d_recv_buffer;

            int *prefix_send_size;
            int *prefix_recv_size;
            array<int, static_pow3(DIMS_t::value)> send_size;
            array<int, static_pow3(DIMS_t::value)> recv_size;

            int *d_send_size;
            int *d_recv_size;

            void *halo_d;   // pointer to halo descr on device
            void *halo_d_r; // pointer to halo descr on device

          public:
            typedef descriptor_base<HaloExch> base_type;
            typedef typename base_type::pattern_type pattern_type;

            /**
               Type of the computin grid associated to the pattern
             */
            typedef typename pattern_type::grid_type grid_type;

            /**
               Type of the translation used to map dimensions to buffer addresses
             */
            typedef translate_t<DIMS_t::value> translate;

            hndlr_generic(grid_type const &g)
                : base_type(g), send_buffer{nullptr}, recv_buffer{nullptr}, send_buffer_size{0}, recv_buffer_size{0} {}

            ~hndlr_generic() {
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk) {
                            gcl_alloc<char, arch_type>::free(send_buffer[translate()(ii, jj, kk)]);
                            gcl_alloc<char, arch_type>::free(recv_buffer[translate()(ii, jj, kk)]);
                        }
                delete[] prefix_send_size;
                delete[] prefix_recv_size;

                GT_CUDA_CHECK(cudaFree(d_send_buffer));
                GT_CUDA_CHECK(cudaFree(d_recv_buffer));
            }

            /**
               function to trigger data exchange

               Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used,
               and vice versa.
            */
            void wait() { base_type::m_haloexch.wait(); }

            /**
               Setup function, in this version, takes tree parameters to
               compute internal buffers and sizes. It takes a field on the fly
               struct, which requires Datatype and layout map template
               arguments that are inferred, so the user is not aware of them.

               \tparam DataType This type is inferred by halo_example paramter
               \tparam data_layout This type is inferred by halo_example paramter

               \param[in] max_fields_n Maximum number of grids used in a computation
               \param[in] halo_example The (at least) maximal grid that is goinf to be used
               \param[in] typesize In case the DataType of the halo_example is not the same as the maximum data type
               used in the computation, this parameter can be given
             */
            template <typename DataType, typename f_data_layout, template <typename> class traits>
            void setup(int max_fields_n,
                field_on_the_fly<DataType, f_data_layout, traits> const &halo_example,
                int typesize = sizeof(DataType)) {
                typedef typename field_on_the_fly<DataType, f_data_layout, traits>::inner_layoutmap data_layout;
                prefix_send_size = new int[max_fields_n * 27];
                prefix_recv_size = new int[max_fields_n * 27];

                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                using map_type = layout_transform<data_layout, proc_layout_abs>;

                                const int ii_P = nth<map_type, 0>(ii, jj, kk);
                                const int jj_P = nth<map_type, 1>(ii, jj, kk);
                                const int kk_P = nth<map_type, 2>(ii, jj, kk);

                                if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                    send_size[translate()(ii, jj, kk)] = halo_example.send_buffer_size({ii, jj, kk});

                                    send_buffer[translate()(ii, jj, kk)] = gcl_alloc<char, arch_type>::alloc(
                                        send_size[translate()(ii, jj, kk)] * max_fields_n * typesize);

                                    base_type::m_haloexch.register_send_to_buffer(
                                        &(send_buffer[translate()(ii, jj, kk)][0]),
                                        send_size[translate()(ii, jj, kk)] * max_fields_n * typesize,
                                        ii_P,
                                        jj_P,
                                        kk_P);

                                    recv_size[translate()(ii, jj, kk)] = halo_example.recv_buffer_size({ii, jj, kk});

                                    recv_buffer[translate()(ii, jj, kk)] = gcl_alloc<char, arch_type>::alloc(
                                        recv_size[translate()(ii, jj, kk)] * max_fields_n * typesize);

                                    base_type::m_haloexch.register_receive_from_buffer(
                                        &(recv_buffer[translate()(ii, jj, kk)][0]),
                                        recv_size[translate()(ii, jj, kk)] * max_fields_n * typesize,
                                        ii_P,
                                        jj_P,
                                        kk_P);

                                } else {
                                    send_size[translate()(ii, jj, kk)] = 0;
                                    send_buffer[translate()(ii, jj, kk)] = nullptr;

                                    base_type::m_haloexch.register_send_to_buffer(nullptr, 0, ii_P, jj_P, kk_P);

                                    recv_size[translate()(ii, jj, kk)] = 0;

                                    recv_buffer[translate()(ii, jj, kk)] = nullptr;

                                    //(*filep) << "Size-of-buffer %d %d %d -> send %d -> recv %d" << ii << jj << kk <<
                                    // send_size[translate()(ii,jj,kk)]*max_fields_n*typesize <<
                                    // recv_size[translate()(ii,jj,kk)]*max_fields_n*typesize << std::endl;
                                    base_type::m_haloexch.register_receive_from_buffer(nullptr, 0, ii_P, jj_P, kk_P);
                                }
                            }

                GT_CUDA_CHECK(cudaMalloc(&d_send_buffer, static_pow3(DIMS_t::value) * sizeof(DataType *)));

                GT_CUDA_CHECK(cudaMemcpy(d_send_buffer,
                    &send_buffer[0],
                    static_pow3(DIMS_t::value) * sizeof(DataType *),
                    cudaMemcpyHostToDevice));

                GT_CUDA_CHECK(cudaMalloc(&d_recv_buffer, static_pow3(DIMS_t::value) * sizeof(DataType *)));

                GT_CUDA_CHECK(cudaMemcpy(d_recv_buffer,
                    &recv_buffer[0],
                    static_pow3(DIMS_t::value) * sizeof(DataType *),
                    cudaMemcpyHostToDevice));
            }

            /**
               Function to unpack received data

               \param[in] _fields vector with data fields pointers to be packed from
            */
            template <typename T1, typename T2, template <typename> class T3>
            void pack(std::vector<field_on_the_fly<T1, T2, T3>> const &_fields) {

                using map_type =
                    layout_transform<typename field_on_the_fly<T1, T2, T3>::inner_layoutmap, proc_layout_abs>;

                std::vector<field_on_the_fly<T1, T2, T3>> fields = _fields;

                {
                    int ii = 1;
                    int jj = 0;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[0].reset_minus();
                    }
                }
                {
                    int ii = -1;
                    int jj = 0;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[0].reset_plus();
                    }
                }
                {
                    int ii = 0;
                    int jj = 1;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[1].reset_minus();
                    }
                }
                {
                    int ii = 0;
                    int jj = -1;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[1].reset_plus();
                    }
                }
                {
                    int ii = 0;
                    int jj = 0;
                    int kk = 1;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[2].reset_minus();
                    }
                }
                {
                    int ii = 0;
                    int jj = 0;
                    int kk = -1;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[2].reset_plus();
                    }
                }

                /* Computing the (prefix sums for) offsets to place fields in linear buffers
                 */
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk) {
                            const int ii_P = nth<map_type, 0>(ii, jj, kk);
                            const int jj_P = nth<map_type, 1>(ii, jj, kk);
                            const int kk_P = nth<map_type, 2>(ii, jj, kk);
                            if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                if (ii != 0 || jj != 0 || kk != 0) {
                                    prefix_send_size[0 + translate()(ii, jj, kk)] = 0;
                                    for (int l = 1; l < fields.size(); ++l) {
                                        prefix_send_size[l * 27 + translate()(ii, jj, kk)] =
                                            prefix_send_size[(l - 1) * 27 + translate()(ii, jj, kk)] +
                                            fields[l - 1].send_buffer_size({ii, jj, kk});
                                    }
                                }
                            }
                        }

                if (send_size[translate()(0, 0, -1)]) {
                    m_packZL_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_send_buffer),
                        &(prefix_send_size[0]));
                }
                if (send_size[translate()(0, 0, 1)]) {
                    m_packZU_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_send_buffer),
                        &(prefix_send_size[0]));
                }
                if (send_size[translate()(0, -1, 0)]) {
                    m_packYL_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_send_buffer),
                        &(prefix_send_size[0]));
                }
                if (send_size[translate()(0, 1, 0)]) {
                    m_packYU_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_send_buffer),
                        &(prefix_send_size[0]));
                }
                if (send_size[translate()(-1, 0, 0)]) {
                    m_packXL_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_send_buffer),
                        &(prefix_send_size[0]));
                }
                if (send_size[translate()(1, 0, 0)]) {
                    m_packXU_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_send_buffer),
                        &(prefix_send_size[0]));
                }

                GT_CUDA_CHECK(cudaDeviceSynchronize());
            }

            /**
               Function to unpack received data

               \param[in] _fields vector with data fields pointers to be unpacked into
            */
            template <typename T1, typename T2, template <typename> class T3>
            void unpack(std::vector<field_on_the_fly<T1, T2, T3>> const &_fields) {
                using map_type =
                    layout_transform<typename field_on_the_fly<T1, T2, T3>::inner_layoutmap, proc_layout_abs>;

                std::vector<field_on_the_fly<T1, T2, T3>> fields = _fields;

                {
                    int ii = 1;
                    int jj = 0;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[0].reset_plus();
                    }
                }
                {
                    int ii = -1;
                    int jj = 0;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[0].reset_minus();
                    }
                }
                {
                    int ii = 0;
                    int jj = 1;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[1].reset_plus();
                    }
                }
                {
                    int ii = 0;
                    int jj = -1;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[1].reset_minus();
                    }
                }
                {
                    int ii = 0;
                    int jj = 0;
                    int kk = 1;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[2].reset_plus();
                    }
                }
                {
                    int ii = 0;
                    int jj = 0;
                    int kk = -1;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        for (int l = 0; l < fields.size(); ++l)
                            fields[l].halos[2].reset_minus();
                    }
                }

                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk) {
                            const int ii_P = nth<map_type, 0>(ii, jj, kk);
                            const int jj_P = nth<map_type, 1>(ii, jj, kk);
                            const int kk_P = nth<map_type, 2>(ii, jj, kk);
                            if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                if (ii != 0 || jj != 0 || kk != 0) {
                                    prefix_recv_size[0 + translate()(ii, jj, kk)] = 0;
                                    for (int l = 1; l < fields.size(); ++l) {
                                        prefix_recv_size[l * 27 + translate()(ii, jj, kk)] =
                                            prefix_recv_size[(l - 1) * 27 + translate()(ii, jj, kk)] +
                                            fields[l - 1].recv_buffer_size({ii, jj, kk});
                                    }
                                }
                            }
                        }

                if (recv_size[translate()(0, 0, -1)]) {
                    m_unpackZL_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_recv_buffer),
                        &(prefix_recv_size[0]));
                }
                if (recv_size[translate()(0, 0, 1)]) {
                    m_unpackZU_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_recv_buffer),
                        &(prefix_recv_size[0]));
                }
                if (recv_size[translate()(0, -1, 0)]) {
                    m_unpackYL_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_recv_buffer),
                        &(prefix_recv_size[0]));
                }
                if (recv_size[translate()(0, 1, 0)]) {
                    m_unpackYU_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_recv_buffer),
                        &(prefix_recv_size[0]));
                }
                if (recv_size[translate()(-1, 0, 0)]) {
                    m_unpackXL_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_recv_buffer),
                        &(prefix_recv_size[0]));
                }
                if (recv_size[translate()(1, 0, 0)]) {
                    m_unpackXU_generic(fields,
                        reinterpret_cast<typename field_on_the_fly<T1, T2, T3>::value_type **>(d_recv_buffer),
                        &(prefix_recv_size[0]));
                }
            }
#include "non_vect_interface.hpp"
        };
#endif // cudacc
    }  // namespace gcl
} // namespace gridtools
