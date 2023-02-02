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

#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "../../sid/blocked_dim.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/sid_shift_origin.hpp"
#include "../common/dim.hpp"
#include "../common/extent.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            namespace tmp_impl_ {
                template <class Extent>
                hymap::keys<dim::i, dim::j, dim::k>::values<integral_constant<int_t, -Extent::iminus::value>,
                    integral_constant<int_t, -Extent::jminus::value>,
                    integral_constant<int_t, -Extent::kminus::value>>
                origin_offset(Extent) {
                    return {};
                }

                template <class NumColors, class IBlockSize, class JBlockSize, class Extent>
                auto sizes(NumColors num_colors,
                    IBlockSize i_block_size,
                    JBlockSize j_block_size,
                    Extent,
                    int_t n_blocks_i,
                    int_t n_blocks_j,
                    int_t k_size) {
                    return hymap::
                        keys<dim::i, dim::j, dim::c, sid::blocked_dim<dim::i>, sid::blocked_dim<dim::j>, dim::k>::
                            make_values(Extent::extend(dim::i(), i_block_size),
                                Extent::extend(dim::j(), j_block_size),
                                num_colors,
                                n_blocks_i,
                                n_blocks_j,
                                Extent::extend(dim::k(), k_size));
                }
            } // namespace tmp_impl_

            // TODO(anstaf): do alignment and padding here.
            template <class Data, class NumColors, class BlockSizeI, class BlockSizeJ, class Extent, class Allocator>
            auto make_tmp_storage(NumColors num_colors,
                BlockSizeI block_size_i,
                BlockSizeJ block_size_j,
                Extent extent,
                int_t n_blocks_i,
                int_t n_blocks_j,
                int_t k_size,
                Allocator &alloc) {
                static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                return sid::shift_sid_origin(
                    sid::make_contiguous<Data, int_t>(alloc,
                        tmp_impl_::sizes(
                            num_colors, block_size_i, block_size_j, extent, n_blocks_i, n_blocks_j, k_size)),
                    tmp_impl_::origin_offset(extent));
            }
        } // namespace gpu_backend
    }     // namespace stencil
} // namespace gridtools
