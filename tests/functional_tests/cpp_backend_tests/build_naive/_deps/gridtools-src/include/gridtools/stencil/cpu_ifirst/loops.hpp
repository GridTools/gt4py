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

#include <type_traits>
#include <utility>

#include "../../common/defs.hpp"
#include "../../common/for_each.hpp"
#include "../../common/omp.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../sid/concept.hpp"
#include "../../thread_pool/concept.hpp"
#include "../common/dim.hpp"
#include "execinfo.hpp"

namespace gridtools {
    namespace stencil {
        namespace cpu_ifirst_backend {
            namespace loops_impl_ {
                template <class Stage, class Ptr, class Strides>
                GT_FORCE_INLINE void i_loop(int_t size, Stage stage, Ptr &ptr, Strides const &strides) {
#pragma omp simd
                    for (int_t i = 0; i < size; ++i) {
                        using namespace literals;
                        stage(ptr, strides);
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
                    }
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), -size);
                }

                template <class Ptr, class Strides>
                struct k_i_loops_f {
                    int_t m_i_size;
                    Ptr &m_ptr;
                    Strides const &m_strides;

                    template <class Cell, class KSize>
                    GT_FORCE_INLINE void operator()(Cell cell, KSize k_size) const {
                        for (int_t k = 0; k < k_size; ++k) {
                            i_loop(m_i_size, cell, m_ptr, m_strides);
                            cell.inc_k(m_ptr, m_strides);
                        }
                    }
                };

                template <class Ptr, class Strides>
                GT_FORCE_INLINE k_i_loops_f<Ptr, Strides> make_k_i_loops(
                    int_t i_size, Ptr &ptr, Strides const &strides) {
                    return {i_size, ptr, strides};
                }

                template <class ThreadPool, class Stage, class Grid, class Composite, class KSizes>
                auto make_loop(std::true_type, Grid const &grid, Composite composite, KSizes k_sizes) {
                    using extent_t = typename Stage::extent_t;
                    using ptr_diff_t = sid::ptr_diff_type<Composite>;
                    auto strides = sid::get_strides(composite);
                    ptr_diff_t offset{};
                    sid::shift(offset, sid::get_stride<dim::i>(strides), extent_t::minus(dim::i()));
                    sid::shift(offset, sid::get_stride<dim::j>(strides), extent_t::minus(dim::j()));
                    return [origin = sid::get_origin(composite) + offset,
                               strides = std::move(strides),
                               k_start = grid.k_start(Stage::interval()),
                               k_sizes = std::move(k_sizes)](execinfo_block_kparallel const &info) {
                        ptr_diff_t offset{};
                        sid::shift(
                            offset, sid::get_stride<dim::thread>(strides), thread_pool::get_thread_num(ThreadPool()));
                        sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), info.i_block);
                        sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), info.j_block);
                        sid::shift(offset, sid::get_stride<dim::k>(strides), info.k);
                        auto ptr = origin() + offset;

                        int_t j_count = extent_t::extend(dim::j(), info.j_block_size);
                        int_t i_size = extent_t::extend(dim::i(), info.i_block_size);

                        for (int_t j = 0; j < j_count; ++j) {
                            using namespace literals;
                            int_t cur = k_start;
                            tuple_util::for_each(
                                [&ptr, &strides, &cur, k = info.k, i_size](auto cell, auto k_size) {
                                    if (k >= cur && k < cur + k_size)
                                        i_loop(i_size, cell, ptr, strides);
                                    cur += k_size;
                                },
                                Stage::cells(),
                                k_sizes);
                            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
                        }
                    };
                }

                template <class ThreadPool, class Grid, class Loops>
                void run_loops(std::true_type, Grid const &grid, Loops loops) {
                    execinfo info(ThreadPool(), grid);
                    int_t i_blocks = info.i_blocks();
                    int_t j_blocks = info.j_blocks();
                    int_t k_size = grid.k_size();
                    thread_pool::parallel_for_loop(
                        ThreadPool(),
                        [&](auto i, auto k, auto j) {
                            tuple_util::for_each([block = info.block(i, j, k)](auto &&loop) { loop(block); }, loops);
                        },
                        i_blocks,
                        k_size,
                        j_blocks);
                }

                template <class ThreadPool, class Stage, class Grid, class Composite, class KSizes>
                auto make_loop(std::false_type, Grid const &grid, Composite composite, KSizes k_sizes) {
                    using extent_t = typename Stage::extent_t;
                    using ptr_diff_t = sid::ptr_diff_type<Composite>;

                    auto strides = sid::get_strides(composite);
                    ptr_diff_t offset{};
                    sid::shift(offset, sid::get_stride<dim::i>(strides), extent_t::minus(dim::i()));
                    sid::shift(offset, sid::get_stride<dim::j>(strides), extent_t::minus(dim::j()));
                    sid::shift(
                        offset, sid::get_stride<dim::k>(strides), grid.k_start(Stage::interval(), Stage::execution()));

                    return [origin = sid::get_origin(composite) + offset,
                               strides = std::move(strides),
                               k_shift_back = -grid.k_size(Stage::interval()) * Stage::k_step(),
                               k_sizes = std::move(k_sizes)](execinfo_block_kserial const &info) {
                        sid::ptr_diff_type<Composite> offset{};
                        sid::shift(
                            offset, sid::get_stride<dim::thread>(strides), thread_pool::get_thread_num(ThreadPool()));
                        sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), info.i_block);
                        sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), info.j_block);
                        auto ptr = origin() + offset;

                        int_t j_size = extent_t::extend(dim::j(), info.j_block_size);
                        int_t i_size = extent_t::extend(dim::i(), info.i_block_size);

                        auto k_i_loops = make_k_i_loops(i_size, ptr, strides);
                        for (int_t j = 0; j < j_size; ++j) {
                            using namespace literals;
                            tuple_util::for_each(k_i_loops, Stage::cells(), k_sizes);
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), k_shift_back);
                            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
                        }
                    };
                }

                template <class ThreadPool, class Grid, class Loops>
                void run_loops(std::false_type, Grid const &grid, Loops loops) {
                    execinfo info(ThreadPool(), grid);
                    thread_pool::parallel_for_loop(
                        ThreadPool(),
                        [&](auto i, auto j) {
                            tuple_util::for_each([block = info.block(i, j)](auto &&loop) { loop(block); }, loops);
                        },
                        info.i_blocks(),
                        info.j_blocks());
                }
            } // namespace loops_impl_
            using loops_impl_::make_loop;
            using loops_impl_::run_loops;
        } // namespace cpu_ifirst_backend
    }     // namespace stencil
} // namespace gridtools
