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
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../meta.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/sid_shift_origin.hpp"
#include "../common/dim.hpp"
#include "../common/extent.hpp"
#include "shared_allocator.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            namespace ij_cache_impl_ {
                template <class Extent>
                auto origin_offset(Extent) {
                    return hymap::keys<dim::i, dim::j>::make_values(-Extent::minus(dim::i()), -Extent::minus(dim::j()));
                }

                template <class NumColors, class IBlockSize, class JBlockSize, class Extent>
                auto sizes(NumColors num_colors, IBlockSize i_block_size, JBlockSize j_block_size, Extent) {
                    return hymap::keys<dim::c, dim::i, dim::j>::make_values(
                        num_colors, Extent::extend(dim::i(), i_block_size), Extent::extend(dim::j(), j_block_size));
                }

                template <class T, class NumColors, class BlockSizeI, class BlockSizeJ, class Extent>
                auto make_ij_cache(NumColors num_colors,
                    BlockSizeI block_size_i,
                    BlockSizeJ block_size_j,
                    Extent extent,
                    shared_allocator &alloc) {
                    static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                    return sid::shift_sid_origin(
                        sid::make_contiguous<T, int_t>(alloc, sizes(num_colors, block_size_i, block_size_j, extent)),
                        origin_offset(extent));
                }
            } // namespace ij_cache_impl_

            using ij_cache_impl_::make_ij_cache;
        } // namespace gpu_backend
    }     // namespace stencil
} // namespace gridtools
