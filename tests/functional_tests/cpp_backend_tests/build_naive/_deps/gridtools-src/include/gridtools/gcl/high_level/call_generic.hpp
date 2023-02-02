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

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include "gcl_parameters.hpp"

// clang-format off
#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, GCL_MAX_FIELDS, <gridtools/gcl/high_level/call_generic.hpp>))
// clang-format on
#include BOOST_PP_ITERATE()

#else

#define GCL_NOI BOOST_PP_ITERATION()

#define _GCL_PACK_F_NAME(x) m_pack##x##_generic_nv
#define GCL_PACK_F_NAME(x) _GCL_PACK_F_NAME(x)

#define _GCL_PACK_FILE_NAME(x) invoke_kernels_##x##_PP.hpp
#define GCL_PACK_FILE_NAME(x) _GCL_PACK_FILE_NAME(x)

#define _GCL_PRINT_FIELDS(z, m, s) \
    (*filep) << "fieldx " << field##m << "\n" << sizeof(typename FOTF_T##m::value_type) << std::endl;
#define GCL_PRINT_FIELDS(m) BOOST_PP_REPEAT(m, _GCL_PRINT_FIELDS, nil)

template <BOOST_PP_ENUM_PARAMS(GCL_NOI, typename FOTF_T)>
void GCL_PACK_F_NAME(GCL_KERNEL_TYPE)(
    BOOST_PP_ENUM_BINARY_PARAMS(GCL_NOI, FOTF_T, const &field), void **d_msgbufTab, const int *d_msgsize) {
    // GCL_PRINT_FIELDS(GCL_NOI);

#define GCL_QUOTE(x) #x
#define _GCL_QUOTE(x) GCL_QUOTE(x)
#include _GCL_QUOTE(GCL_PACK_FILE_NAME(GCL_KERNEL_TYPE))
#undef GCL_QUOTE
#undef _GCL_QUOTE
}

#define _GCL_UNPACK_F_NAME(x) m_unpack##x##_generic_nv
#define GCL_UNPACK_F_NAME(x) _GCL_UNPACK_F_NAME(x)

#define _GCL_UNPACK_FILE_NAME(x) invoke_kernels_U_##x##_PP.hpp
#define GCL_UNPACK_FILE_NAME(x) _GCL_UNPACK_FILE_NAME(x)

template <BOOST_PP_ENUM_PARAMS(GCL_NOI, typename FOTF_T)>
void GCL_UNPACK_F_NAME(GCL_KERNEL_TYPE)(
    BOOST_PP_ENUM_BINARY_PARAMS(GCL_NOI, FOTF_T, const &field), void **d_msgbufTab_r, int *d_msgsize_r) {

#define GCL_QUOTE(x) #x
#define _GCL_QUOTE(x) GCL_QUOTE(x)
#include _GCL_QUOTE(GCL_UNPACK_FILE_NAME(GCL_KERNEL_TYPE))
#undef GCL_QUOTE
#undef _GCL_QUOTE
}

#undef GCL_PACK_F_NAME
#undef _GCL_PACK_F_NAME
#undef GCL_PACK_FILE_NAME
#undef _GCL_PACK_FILE_NAME
#undef GCL_UNPACK_F_NAME
#undef _GCL_UNPACK_F_NAME
#undef GCL_UNPACK_FILE_NAME
#undef _GCL_UNPACK_FILE_NAME
#undef GCL_PRINT_FIELDS
#undef _GCL_PRINT_FIELDS
#undef GCL_NOI

#endif
