/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#if !BOOST_PP_IS_ITERATING

#ifndef GCL_NON_VECT_INTERFACE_H_
#define GCL_NON_VECT_INTERFACE_H_

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include "gcl_parameters.hpp"

#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, GCL_MAX_FIELDS, "gridtools/gcl/high_level/non_vect_interface.hpp"))
#include BOOST_PP_ITERATE()

#endif
#else

#define GCL_NOI BOOST_PP_ITERATION()

// #define _GCL_TRIM_FIELDS(z, m, s) field ## m
// #define GCL_TRIM_FIELDS(m, s) BOOST_PP_REPEAT(m, _GCL_TRIM_FIELDS, s)

#define _GCL_PRINT_FIELDS(z, m, s) (*filep) << _field##m << "\n" << sizeof(FIELD##m) << std::endl;
// std::cout << fields[ m ] << " is equal to (input) " << _field ## m << std::endl;
#define GCL_PRINT_FIELDS(m) BOOST_PP_REPEAT(m, _GCL_PRINT_FIELDS, nil)

#define _GCL_COPY_FIELDS(z, m, s) fields[m] = (_field##m.template copy<typename FIELD0::value_type>());
// std::cout << fields[ m ] << " is equal to (input) " << _field ## m << std::endl;
#define GCL_COPY_FIELDS(m) BOOST_PP_REPEAT(m, _GCL_COPY_FIELDS, nil)

#define _GCL_COPY_BACK(z, m, s) FIELD##m &new_field##m = fields[m].template retarget<typename FIELD##m::value_type>();
#define GCL_COPY_BACK(m) BOOST_PP_REPEAT(m, _GCL_COPY_BACK, nil)

template <BOOST_PP_ENUM_PARAMS(GCL_NOI, typename FIELD)>
void pack(BOOST_PP_ENUM_BINARY_PARAMS(GCL_NOI, FIELD, const &_field)) const {
    ////////////////////////////////// Only FIELD0 is taken for layout... all should have the same
    using map_type = layout_transform<typename FIELD0::inner_layoutmap, proc_layout_abs>;

    std::vector<FIELD0> fields(GCL_NOI);

    GCL_COPY_FIELDS(GCL_NOI);

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

    // for (int l=0; l<fields.size(); ++l)
    //   std::cout << "after trimming " << l << " " << fields[l] << std::endl;

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
                        //(*filep) << "prefix_send_size[l*27+translate()(ii,jj,kk)]=prefix_send_size[%d]=%d\n" <<
                        // 0*27+translate()(ii,jj,kk) << prefix_send_size[0*27+translate()(ii,jj,kk)] << std::endl;
                        for (int l = 1; l < fields.size(); ++l) {
                            prefix_send_size[l * 27 + translate()(ii, jj, kk)] =
                                prefix_send_size[(l - 1) * 27 + translate()(ii, jj, kk)] +
                                fields[l - 1].send_buffer_size({ii, jj, kk});
                            //(*filep) << "prefix_send_size[l*27+translate()(ii,jj,kk)]=prefix_send_size[%d]=%d\n" <<
                            // l*27+translate()(ii,jj,kk) << prefix_send_size[l*27+translate()(ii,jj,kk)] << std::endl;
                        }
                    }
                }
            }

    assert(fields.size() == GCL_NOI);

    GCL_COPY_BACK(GCL_NOI);

    if (send_size[translate()(0, 0, -1)]) {
        m_packZL_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_send_buffer), prefix_send_size);
    }
    if (send_size[translate()(0, 0, 1)]) {
        m_packZU_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_send_buffer), prefix_send_size);
    }
    if (send_size[translate()(0, -1, 0)]) {
        m_packYL_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_send_buffer), prefix_send_size);
    }
    if (send_size[translate()(0, 1, 0)]) {
        m_packYU_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_send_buffer), prefix_send_size);
    }
    if (send_size[translate()(-1, 0, 0)]) {
        m_packXL_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_send_buffer), prefix_send_size);
    }
    if (send_size[translate()(1, 0, 0)]) {
        m_packXU_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_send_buffer), prefix_send_size);
    }

    GT_CUDA_CHECK(cudaDeviceSynchronize());
}

/**
   Function to unpack received data

   \param[in] fields vector with data fields pointers to be unpacked into
*/
template <BOOST_PP_ENUM_PARAMS(GCL_NOI, typename FIELD)>
void unpack(BOOST_PP_ENUM_BINARY_PARAMS(GCL_NOI, FIELD, const &_field)) const {
    ////////////////////////////////// Only FIELD0 is taken for layout... all should have the same
    using map_type = layout_transform<typename FIELD0::inner_layoutmap, proc_layout_abs>;

    std::vector<FIELD0> fields(GCL_NOI);

    GCL_COPY_FIELDS(GCL_NOI)

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

    GCL_COPY_BACK(GCL_NOI);

    if (recv_size[translate()(0, 0, -1)]) {
        m_unpackZL_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_recv_buffer), prefix_recv_size);
    }
    if (recv_size[translate()(0, 0, 1)]) {
        m_unpackZU_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_recv_buffer), prefix_recv_size);
    }
    if (recv_size[translate()(0, -1, 0)]) {
        m_unpackYL_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_recv_buffer), prefix_recv_size);
    }
    if (recv_size[translate()(0, 1, 0)]) {
        m_unpackYU_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_recv_buffer), prefix_recv_size);
    }
    if (recv_size[translate()(-1, 0, 0)]) {
        m_unpackXL_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_recv_buffer), prefix_recv_size);
    }
    if (recv_size[translate()(1, 0, 0)]) {
        m_unpackXU_generic_nv(
            BOOST_PP_ENUM_PARAMS(GCL_NOI, new_field), reinterpret_cast<void **>(d_recv_buffer), prefix_recv_size);
    }
}

#undef GCL_NOI

#undef _GCL_TRIM_FIELDS
#undef GCL_TRIM_FIELDS

#undef _GCL_COPY_FIELDS
#undef GCL_COPY_FIELDS

#undef _GCL_PREFIX_SEND
#undef GCL_PREFIX_SEND

#undef _GCL_COPY_BACK
#undef GCL_COPY_BACK

#endif
