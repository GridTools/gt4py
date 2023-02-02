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

#include "../../common/defs.hpp"
#include "empty_field_base.hpp"
#include "helpers_impl.hpp"
#include "numerics.hpp"

#ifdef GT_CUDACC
#include "m_packXL.hpp"
#include "m_packXU.hpp"
#include "m_packYL.hpp"
#include "m_packYU.hpp"
#include "m_packZL.hpp"
#include "m_packZU.hpp"

#include "m_unpackXL.hpp"
#include "m_unpackXU.hpp"
#include "m_unpackYL.hpp"
#include "m_unpackYU.hpp"
#include "m_unpackZL.hpp"
#include "m_unpackZU.hpp"
#endif

namespace gridtools {
    namespace gcl {
        /** \class empty_field_no_dt_gpu
            Class containint the information about a data field (grid).
            It doe not contains any reference to actual data of the field,
            it only describes the fields though the halo descriptions.
            The number of dimensions as a template argument and the size of the
            first dimension, the size of the non-halo data field,
            the halo width before and after the actual data, then the same for the
            second dimension, the third, etc. This information is encoded in
            halo_descriptor. A dimension of the field is described as:
            \code
            |-----|------|---------------|---------|----|
            | pad0|minus |    length     | plus    |pad1|
                          ^begin        ^end
            |               total_length                |
            \endcode

            \tparam DIMS the number of dimensions of the data field
        */
        template <int DIMS>
        class empty_field_no_dt_gpu : public empty_field_base<int> {

            typedef empty_field_base<int> base_type;

          public:
            /**
                Constructor that receive the pointer to the data. This is explicit and
                must then be called.
            */
            explicit empty_field_no_dt_gpu() {}

            void setup() const {}

            const halo_descriptor *raw_array() const { return &(base_type::halos[0]); }

          protected:
            template <typename DataType, typename GridType, typename HaloExch, typename proc_layout, typename arch>
            friend class hndlr_dynamic_ut;

            template <int I>
            friend std::ostream &operator<<(std::ostream &s, empty_field_no_dt_gpu<I> const &ef);

            halo_descriptor *dangerous_raw_array() { return &(base_type::halos[0]); }
        };

#ifdef GT_CUDACC
        /** specialization for GPU and manual packing */
        template <typename DataType, typename HaloExch, typename proc_layout, template <int Ndim> class GridType>
        class hndlr_dynamic_ut<DataType, GridType<3>, HaloExch, proc_layout, gpu> : public descriptor_base<HaloExch> {

            static const int DIMS = 3;

            typedef hndlr_dynamic_ut<DataType, GridType<3>, HaloExch, proc_layout, gpu> this_type;

          public:
            empty_field_no_dt_gpu<DIMS> halo;

          private:
            typedef gpu arch_type;
            DataType **d_send_buffer;
            DataType **d_recv_buffer;

            halo_descriptor dangeroushalo[3];
            halo_descriptor dangeroushalo_r[3];
            array<DataType *, static_pow3(DIMS)> send_buffer;
            array<DataType *, static_pow3(DIMS)> recv_buffer;
            array<int, static_pow3(DIMS)> send_size;
            array<int, static_pow3(DIMS)> recv_size;
            int *d_send_size;
            int *d_recv_size;

            halo_descriptor *halo_d;   // pointer to halo descr on device
            halo_descriptor *halo_d_r; // pointer to halo descr on device
          public:
            typedef descriptor_base<HaloExch> base_type;
            typedef base_type pattern_type;

            /**
               Type of the computin grid associated to the pattern
             */
            typedef typename pattern_type::grid_type grid_type;

            /**
               Type of the translation used to map dimensions to buffer addresses
             */
            typedef translate_t<DIMS> translate;

          private:
            hndlr_dynamic_ut(hndlr_dynamic_ut const &) = delete;
            hndlr_dynamic_ut(hndlr_dynamic_ut &&) = delete;

          public:
            /**
               Constructor

               \param[in] c The object of the class used to specify periodicity in each dimension
               \param[in] comm MPI communicator (typically MPI_Comm_world)
            */
            explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, MPI_Comm const &comm)
                : base_type(c, comm), send_buffer{nullptr}, recv_buffer{nullptr}, send_size{0}, recv_size{0} {}

            /**
               Constructor

               \param[in] g A processor grid that will execute the pattern
             */
            explicit hndlr_dynamic_ut(grid_type const &g)
                : base_type(g), send_buffer{nullptr}, recv_buffer{nullptr}, send_size{0}, recv_size{0} {}

            ~hndlr_dynamic_ut() {
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        for (int k = -1; k <= 1; ++k) {
                            gcl_alloc<DataType, arch_type>::free(send_buffer[translate()(i, j, k)]);
                            gcl_alloc<DataType, arch_type>::free(recv_buffer[translate()(i, j, k)]);
                        }
                GT_CUDA_CHECK(cudaFree(d_send_buffer));
                GT_CUDA_CHECK(cudaFree(d_recv_buffer));
                GT_CUDA_CHECK(cudaFree(d_send_size));
                GT_CUDA_CHECK(cudaFree(d_recv_size));
                GT_CUDA_CHECK(cudaFree(halo_d));
                GT_CUDA_CHECK(cudaFree(halo_d_r));
            }

            /**
               Function to setup internal data structures for data exchange and preparing eventual underlying layers

               \param max_fields_n Maximum number of data fields that will be passed to the communication functions
            */
            void setup(const int max_fields_n) {

                typedef translate_t<3> translate;
                typedef translate_t<3, proc_layout> translate_P;

                dangeroushalo[0] = halo.dangerous_raw_array()[0];
                dangeroushalo[1] = halo.dangerous_raw_array()[1];
                dangeroushalo[2] = halo.dangerous_raw_array()[2];
                dangeroushalo_r[0] = halo.dangerous_raw_array()[0];
                dangeroushalo_r[1] = halo.dangerous_raw_array()[1];
                dangeroushalo_r[2] = halo.dangerous_raw_array()[2];

                {
                    typedef proc_layout map_type;
                    int ii = 1;
                    int jj = 0;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        dangeroushalo[0].reset_minus();
                        dangeroushalo_r[0].reset_plus();
                    }
                }
                {
                    typedef proc_layout map_type;
                    int ii = -1;
                    int jj = 0;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        dangeroushalo[0].reset_plus();
                        dangeroushalo_r[0].reset_minus();
                    }
                }
                {
                    typedef proc_layout map_type;
                    int ii = 0;
                    int jj = 1;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        dangeroushalo[1].reset_minus();
                        dangeroushalo_r[1].reset_plus();
                    }
                }
                {
                    typedef proc_layout map_type;
                    int ii = 0;
                    int jj = -1;
                    int kk = 0;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        dangeroushalo[1].reset_plus();
                        dangeroushalo_r[1].reset_minus();
                    }
                }
                {
                    typedef proc_layout map_type;
                    int ii = 0;
                    int jj = 0;
                    int kk = 1;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        dangeroushalo[2].reset_minus();
                        dangeroushalo_r[2].reset_plus();
                    }
                }
                {
                    typedef proc_layout map_type;
                    int ii = 0;
                    int jj = 0;
                    int kk = -1;
                    const int ii_P = nth<map_type, 0>(ii, jj, kk);
                    const int jj_P = nth<map_type, 1>(ii, jj, kk);
                    const int kk_P = nth<map_type, 2>(ii, jj, kk);
                    if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                        dangeroushalo[2].reset_plus();
                        dangeroushalo_r[2].reset_minus();
                    }
                }

                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                typedef typename translate_P::map_type map_type;
                                const int ii_P = nth<map_type, 0>(ii, jj, kk);
                                const int jj_P = nth<map_type, 1>(ii, jj, kk);
                                const int kk_P = nth<map_type, 2>(ii, jj, kk);

                                if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                    send_size[translate()(ii, jj, kk)] = halo.send_buffer_size({ii, jj, kk});

                                    send_buffer[translate()(ii, jj, kk)] = gcl_alloc<DataType, arch_type>::alloc(
                                        send_size[translate()(ii, jj, kk)] * max_fields_n);

                                    base_type::m_haloexch.register_send_to_buffer(
                                        &(send_buffer[translate()(ii, jj, kk)][0]),
                                        send_size[translate()(ii, jj, kk)] * max_fields_n * sizeof(DataType),
                                        ii_P,
                                        jj_P,
                                        kk_P);

                                    recv_size[translate()(ii, jj, kk)] = halo.recv_buffer_size({ii, jj, kk});

                                    recv_buffer[translate()(ii, jj, kk)] = gcl_alloc<DataType, arch_type>::alloc(
                                        recv_size[translate()(ii, jj, kk)] * max_fields_n);

                                    base_type::m_haloexch.register_receive_from_buffer(
                                        &(recv_buffer[translate()(ii, jj, kk)][0]),
                                        recv_size[translate()(ii, jj, kk)] * max_fields_n * sizeof(DataType),
                                        ii_P,
                                        jj_P,
                                        kk_P);
                                } else {
                                    send_size[translate()(ii, jj, kk)] = 0;
                                    send_buffer[translate()(ii, jj, kk)] = nullptr;

                                    base_type::m_haloexch.register_send_to_buffer(nullptr, 0, ii_P, jj_P, kk_P);

                                    recv_size[translate()(ii, jj, kk)] = 0;

                                    recv_buffer[translate()(ii, jj, kk)] = nullptr;

                                    base_type::m_haloexch.register_receive_from_buffer(nullptr, 0, ii_P, jj_P, kk_P);
                                }
                            }

                GT_CUDA_CHECK(cudaMalloc(&d_send_buffer, static_pow3(DIMS) * sizeof(DataType *)));

                GT_CUDA_CHECK(cudaMemcpy(
                    d_send_buffer, &send_buffer[0], static_pow3(DIMS) * sizeof(DataType *), cudaMemcpyHostToDevice));

                GT_CUDA_CHECK(cudaMalloc(&d_recv_buffer, static_pow3(DIMS) * sizeof(DataType *)));

                GT_CUDA_CHECK(cudaMemcpy(
                    d_recv_buffer, &recv_buffer[0], static_pow3(DIMS) * sizeof(DataType *), cudaMemcpyHostToDevice));

                GT_CUDA_CHECK(cudaMalloc(&d_send_size, static_pow3(DIMS) * sizeof(int)));

                GT_CUDA_CHECK(
                    cudaMemcpy(d_send_size, &send_size[0], static_pow3(DIMS) * sizeof(int), cudaMemcpyHostToDevice));

                GT_CUDA_CHECK(cudaMalloc(&d_recv_size, static_pow3(DIMS) * sizeof(int)));

                GT_CUDA_CHECK(
                    cudaMemcpy(d_recv_size, &recv_size[0], static_pow3(DIMS) * sizeof(int), cudaMemcpyHostToDevice));

                GT_CUDA_CHECK(cudaMalloc(&halo_d, DIMS * sizeof(halo_descriptor)));

                GT_CUDA_CHECK(cudaMemcpy(halo_d,
                    dangeroushalo /*halo.raw_array()*/,
                    DIMS * sizeof(halo_descriptor),
                    cudaMemcpyHostToDevice));

                GT_CUDA_CHECK(cudaMalloc(&halo_d_r, DIMS * sizeof(halo_descriptor)));

                GT_CUDA_CHECK(cudaMemcpy(halo_d_r,
                    dangeroushalo_r /*halo.raw_array()*/,
                    DIMS * sizeof(halo_descriptor),
                    cudaMemcpyHostToDevice));
            }

            /**
               Function to pack data before sending

               \param[in] fields vector with data fields pointers to be packed from
            */
            template <typename... Pointers>
            void pack(const Pointers *...fields) {
                typedef translate_t<3> translate;
                auto ints = std::make_integer_sequence<unsigned int, sizeof...(Pointers)>{};
                if (send_size[translate()(0, 0, -1)]) {
                    m_packZL_variadic(
                        d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
                }

                if (send_size[translate()(0, 0, 1)]) {
                    m_packZU_variadic(
                        d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
                }

                if (send_size[translate()(0, -1, 0)]) {
                    m_packYL_variadic(
                        d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
                }

                if (send_size[translate()(0, 1, 0)]) {
                    m_packYU_variadic(
                        d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
                }

                if (send_size[translate()(-1, 0, 0)]) {
                    m_packXL_variadic(
                        d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
                }

                if (send_size[translate()(1, 0, 0)]) {
                    m_packXU_variadic(
                        d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
                }

                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                using translate_P = translate_t<3, proc_layout>;
                                using map_type = typename translate_P::map_type;
                                const int ii_P = nth<map_type, 0>(ii, jj, kk);
                                const int jj_P = nth<map_type, 1>(ii, jj, kk);
                                const int kk_P = nth<map_type, 2>(ii, jj, kk);

                                if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                    base_type::m_haloexch.set_send_to_size(
                                        send_size[translate()(ii, jj, kk)] * sizeof...(fields) * sizeof(DataType),
                                        ii_P,
                                        jj_P,
                                        kk_P);
                                    base_type::m_haloexch.set_receive_from_size(
                                        recv_size[translate()(ii, jj, kk)] * sizeof...(fields) * sizeof(DataType),
                                        ii_P,
                                        jj_P,
                                        kk_P);
                                }
                            }

                GT_CUDA_CHECK(cudaDeviceSynchronize());
            }

            template <typename... Pointers>
            void unpack(Pointers *...fields) {
                auto ints = std::make_integer_sequence<unsigned int, sizeof...(Pointers)>{};
                typedef translate_t<3> translate;
                if (recv_size[translate()(0, 0, -1)]) {
                    m_unpackZL_variadic(
                        d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
                }
                if (recv_size[translate()(0, 0, 1)]) {
                    m_unpackZU_variadic(
                        d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
                }
                if (recv_size[translate()(0, -1, 0)]) {
                    m_unpackYL_variadic(
                        d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
                }
                if (recv_size[translate()(0, 1, 0)]) {
                    m_unpackYU_variadic(
                        d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
                }
                if (recv_size[translate()(-1, 0, 0)]) {
                    m_unpackXL_variadic(
                        d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
                }
                if (recv_size[translate()(1, 0, 0)]) {
                    m_unpackXU_variadic(
                        d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
                }
            }

            /**
               Function to pack data before sending

               \param[in] fields vector with data fields pointers to be packed from
            */
            void pack(std::vector<DataType *> const &fields) {
                typedef translate_t<3> translate;
                if (send_size[translate()(0, 0, -1)]) {
                    m_packZL(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
                }
                if (send_size[translate()(0, 0, 1)]) {
                    m_packZU(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
                }
                if (send_size[translate()(0, -1, 0)]) {
                    m_packYL(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
                }
                if (send_size[translate()(0, 1, 0)]) {
                    m_packYU(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
                }
                if (send_size[translate()(-1, 0, 0)]) {
                    m_packXL(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
                }
                if (send_size[translate()(1, 0, 0)]) {
                    m_packXU(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
                }

                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                using translate_P = translate_t<3, proc_layout>;
                                using map_type = typename translate_P::map_type;
                                const int ii_P = nth<map_type, 0>(ii, jj, kk);
                                const int jj_P = nth<map_type, 1>(ii, jj, kk);
                                const int kk_P = nth<map_type, 2>(ii, jj, kk);

                                if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                    base_type::m_haloexch.set_send_to_size(
                                        send_size[translate()(ii, jj, kk)] * fields.size() * sizeof(DataType),
                                        ii_P,
                                        jj_P,
                                        kk_P);
                                    base_type::m_haloexch.set_receive_from_size(
                                        recv_size[translate()(ii, jj, kk)] * fields.size() * sizeof(DataType),
                                        ii_P,
                                        jj_P,
                                        kk_P);
                                }
                            }

                // perform device syncronization to ensure that packing is finished
                // before MPI is called with the device pointers, otherwise stale
                // information can be sent
                GT_CUDA_CHECK(cudaDeviceSynchronize());
            }

            /**
               Function to unpack received data

               \param[in] fields vector with data fields pointers to be unpacked into
            */
            void unpack(std::vector<DataType *> const &fields) {
                typedef translate_t<3> translate;
                if (recv_size[translate()(0, 0, -1)]) {
                    m_unpackZL(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
                }
                if (recv_size[translate()(0, 0, 1)]) {
                    m_unpackZU(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
                }
                if (recv_size[translate()(0, -1, 0)]) {
                    m_unpackYL(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
                }
                if (recv_size[translate()(0, 1, 0)]) {
                    m_unpackYU(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
                }
                if (recv_size[translate()(-1, 0, 0)]) {
                    m_unpackXL(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
                }
                if (recv_size[translate()(1, 0, 0)]) {
                    m_unpackXU(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
                }
            }
        };
#endif
    } // namespace gcl
} // namespace gridtools
