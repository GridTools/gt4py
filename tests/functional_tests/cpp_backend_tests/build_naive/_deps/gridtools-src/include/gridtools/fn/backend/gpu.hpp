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

#include <utility>

#include "../../common/cuda_util.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../../sid/allocator.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/multi_shift.hpp"
#include "../../sid/unknown_kind.hpp"
#include "./common.hpp"

namespace gridtools::fn::backend {
    namespace gpu_impl_ {
        /*
         * BlockSizes must be a meta map, mapping dimensions to integral constant block sizes.
         *
         * For example, meta::list<meta::list<dim::i, integral_constant<int, 32>>,
         *                         meta::list<dim::j, integral_constant<int, 8>>,
         *                         meta::list<dim::k, integral_constant<int, 1>>>;
         * When using a cartesian grid.
         */
        template <class BlockSizes>
        struct gpu {
            using block_sizes_t = BlockSizes;
            cudaStream_t stream = 0;
        };

        template <class BlockSizes, class Dims, int I>
        using block_size_at_dim = meta::second<meta::mp_find<BlockSizes, meta::at_c<Dims, I>>>;

        template <class BlockSizes, class Sizes>
        GT_FUNCTION_DEVICE auto global_thread_index() {
            using all_keys_t = get_keys<Sizes>;
            using ndims_t = meta::length<all_keys_t>;
            using keys_t = meta::rename<hymap::keys, meta::take_c<std::min(3, (int)ndims_t::value), all_keys_t>>;
            if constexpr (ndims_t::value == 0) {
                return hymap::keys<>::values<>();
            } else if constexpr (ndims_t::value == 1) {
                using block_dim_x = block_size_at_dim<BlockSizes, keys_t, 0>;
                using values_t = typename keys_t::template values<int>;
                return values_t(blockIdx.x * block_dim_x::value + threadIdx.x);
            } else if constexpr (ndims_t::value == 2) {
                using block_dim_x = block_size_at_dim<BlockSizes, keys_t, 0>;
                using block_dim_y = block_size_at_dim<BlockSizes, keys_t, 1>;
                using values_t = typename keys_t::template values<int, int>;
                return values_t(
                    blockIdx.x * block_dim_x::value + threadIdx.x, blockIdx.y * block_dim_y::value + threadIdx.y);
            } else {
                using block_dim_x = block_size_at_dim<BlockSizes, keys_t, 0>;
                using block_dim_y = block_size_at_dim<BlockSizes, keys_t, 1>;
                using block_dim_z = block_size_at_dim<BlockSizes, keys_t, 2>;
                using values_t = typename keys_t::template values<int, int, int>;
                return values_t(blockIdx.x * block_dim_x::value + threadIdx.x,
                    blockIdx.y * block_dim_y::value + threadIdx.y,
                    blockIdx.z * block_dim_z::value + threadIdx.z);
            }
            // disable incorrect warning "missing return statement at end of non-void function"
            GT_NVCC_DIAG_PUSH_SUPPRESS(940)
        }
        GT_NVCC_DIAG_POP_SUPPRESS(940)

        template <class Key>
        struct at_generator_f {
            template <class Value>
            GT_FUNCTION_DEVICE decltype(auto) operator()(Value &&value) const {
                return device::at_key<Key>(std::forward<Value>(value));
            }
        };

        template <class Index, class Sizes>
        GT_FUNCTION_DEVICE bool in_domain(Index const &index, Sizes const &sizes) {
            using sizes_t = meta::rename<tuple, Index>;
            using generators_t = meta::transform<at_generator_f, get_keys<Index>>;
            auto indexed_sizes = tuple_util::device::generate<generators_t, sizes_t>(sizes);
            return tuple_util::device::all_of(std::less(), index, indexed_sizes);
        }

        template <class BlockSizes,
            class Sizes,
            class PtrHolder,
            class Strides,
            class Fun,
            class NDims = tuple_util::size<Sizes>,
            class SizeKeys = get_keys<Sizes>>
        __global__ void kernel(Sizes sizes, PtrHolder ptr_holder, Strides strides, Fun fun) {
            auto thread_idx = global_thread_index<BlockSizes, Sizes>();
            if (!in_domain(thread_idx, sizes))
                return;
            auto ptr = ptr_holder();
            sid::multi_shift(ptr, strides, thread_idx);
            if constexpr (NDims::value <= 3) {
                fun(ptr, strides);
            } else {
                using loop_dims_t = meta::drop_front_c<3, SizeKeys>;
                common::make_loops<loop_dims_t>(sizes)(std::move(fun))(ptr, strides);
            }
        }

        template <class BlockSizes, class Sizes>
        std::tuple<dim3, dim3> blocks_and_threads(Sizes const &sizes) {
            using keys_t = get_keys<Sizes>;
            using ndims_t = meta::length<keys_t>;
            dim3 blocks(1, 1, 1);
            dim3 threads(1, 1, 1);
            if constexpr (ndims_t::value >= 1) {
                threads.x = block_size_at_dim<BlockSizes, keys_t, 0>();
                blocks.x = (tuple_util::get<0>(sizes) + threads.x - 1) / threads.x;
            }
            if constexpr (ndims_t::value >= 2) {
                threads.y = block_size_at_dim<BlockSizes, keys_t, 1>();
                blocks.y = (tuple_util::get<1>(sizes) + threads.y - 1) / threads.y;
            }
            if constexpr (ndims_t::value >= 3) {
                threads.z = block_size_at_dim<BlockSizes, keys_t, 2>();
                blocks.z = (tuple_util::get<2>(sizes) + threads.z - 1) / threads.z;
            }
            return {blocks, threads};
        }

        template <class StencilStage, class MakeIterator>
        struct stencil_fun_f {
            MakeIterator m_make_iterator;

            template <class Ptr, class Strides>
            GT_FUNCTION_DEVICE void operator()(Ptr &ptr, Strides const &strides) const {
                StencilStage()(m_make_iterator(), ptr, strides);
            }
        };

        template <class Sizes>
        bool is_domain_empty(const Sizes &sizes) {
            return tuple_util::host::apply([](auto... sizes) { return ((sizes == 0) || ...); }, sizes);
        }

        template <class BlockSizes, class Sizes, class StencilStage, class MakeIterator, class Composite>
        void apply_stencil_stage(gpu<BlockSizes> const &g,
            Sizes const &sizes,
            StencilStage,
            MakeIterator make_iterator,
            Composite &&composite) {

            if (is_domain_empty(sizes)) {
                return;
            }

            auto ptr_holder = sid::get_origin(std::forward<Composite>(composite));
            auto strides = sid::get_strides(std::forward<Composite>(composite));

            auto [blocks, threads] = blocks_and_threads<BlockSizes>(sizes);
            assert(threads.x > 0 && threads.y > 0 && threads.z > 0);
            cuda_util::launch(blocks,
                threads,
                0,
                g.stream,
                kernel<BlockSizes,
                    Sizes,
                    decltype(ptr_holder),
                    decltype(strides),
                    stencil_fun_f<StencilStage, MakeIterator>>,
                sizes,
                ptr_holder,
                strides,
                stencil_fun_f<StencilStage, MakeIterator>{std::move(make_iterator)});
        }

        template <class ColumnStage, class MakeIterator, class Seed>
        struct column_fun_f {
            MakeIterator m_make_iterator;
            Seed m_seed;
            int m_v_size;

            template <class Ptr, class Strides>
            GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides) const {
                ColumnStage()(m_seed, m_v_size, m_make_iterator(), std::move(ptr), strides);
            }
        };

        template <class BlockSizes,
            class Sizes,
            class ColumnStage,
            class MakeIterator,
            class Composite,
            class Vertical,
            class Seed>
        void apply_column_stage(gpu<BlockSizes> const &g,
            Sizes const &sizes,
            ColumnStage,
            MakeIterator make_iterator,
            Composite &&composite,
            Vertical,
            Seed seed) {

            if (is_domain_empty(sizes)) {
                return;
            }

            auto ptr_holder = sid::get_origin(std::forward<Composite>(composite));
            auto strides = sid::get_strides(std::forward<Composite>(composite));
            auto h_sizes = hymap::canonicalize_and_remove_key<Vertical>(sizes);
            int v_size = at_key<Vertical>(sizes);

            auto [blocks, threads] = blocks_and_threads<BlockSizes>(h_sizes);
            assert(threads.x > 0 && threads.y > 0 && threads.z > 0);
            cuda_util::launch(blocks,
                threads,
                0,
                g.stream,
                kernel<BlockSizes,
                    decltype(h_sizes),
                    decltype(ptr_holder),
                    decltype(strides),
                    column_fun_f<ColumnStage, MakeIterator, Seed>>,
                h_sizes,
                ptr_holder,
                strides,
                column_fun_f<ColumnStage, MakeIterator, Seed>{std::move(make_iterator), std::move(seed), v_size});
        }

        template <class BlockSizes>
        auto tmp_allocator(gpu<BlockSizes> be) {
            return std::make_tuple(be, sid::device::cached_allocator(&cuda_util::cuda_malloc<char[]>));
        }

        template <class BlockSizes, class Allocator, class Sizes, class T>
        auto allocate_global_tmp(std::tuple<gpu<BlockSizes>, Allocator> &alloc, Sizes const &sizes, data_type<T>) {
            return sid::make_contiguous<T, int_t, sid::unknown_kind>(std::get<1>(alloc), sizes);
        }
    } // namespace gpu_impl_

    using gpu_impl_::gpu;

    using gpu_impl_::apply_column_stage;
    using gpu_impl_::apply_stencil_stage;

    using gpu_impl_::allocate_global_tmp;
    using gpu_impl_::tmp_allocator;
} // namespace gridtools::fn::backend
