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
#include "../../common/host_device.hpp"
#include "../../thread_pool/concept.hpp"

namespace gridtools {
    namespace stencil {
        namespace cpu_ifirst_backend {
            /**
             *  @brief Execution info class for MC backend.
             *  Used for stencils that are executed serially along the k-axis.
             */
            struct execinfo_block_kserial {
                int_t i_block;
                int_t j_block;
                int_t i_block_size; /** Size of block along i-axis. */
                int_t j_block_size; /** Size of block along j-axis. */
            };

            /**
             *  @brief Execution info class for MC backend.
             *  Used for stencils that are executed in parallel along the k-axis.
             */
            struct execinfo_block_kparallel {
                int_t i_block;
                int_t j_block;
                int_t k;            /** Position along k-axis. */
                int_t i_block_size; /** Size of block along i-axis. */
                int_t j_block_size; /** Size of block along j-axis. */
            };

            /**
             * @brief Helper class for block handling.
             */
            class execinfo {
                int_t m_i_grid_size, m_j_grid_size;
                int_t m_i_block_size, m_j_block_size;
                int_t m_i_blocks, m_j_blocks;

                GT_FORCE_INLINE static int_t clamped_block_size(
                    int_t grid_size, int_t block_index, int_t block_size, int_t blocks) {
                    return (block_index == blocks - 1) ? grid_size - block_index * block_size : block_size;
                }

              public:
                template <class ThreadPool, class Grid>
                GT_FORCE_INLINE execinfo(ThreadPool, const Grid &grid)
                    : m_i_grid_size(grid.i_size()), m_j_grid_size(grid.j_size()) {
                    int_t threads = thread_pool::get_max_threads(ThreadPool());

                    // if domain is large enough (relative to the number of threads),
                    // we split only along j-axis (for prefetching reasons)
                    // for smaller domains we also split along i-axis
                    m_j_block_size = (m_j_grid_size + threads - 1) / threads;
                    m_j_blocks = (m_j_grid_size + m_j_block_size - 1) / m_j_block_size;
                    int_t max_i_blocks = threads / m_j_blocks;
                    m_i_block_size = (m_i_grid_size + max_i_blocks - 1) / max_i_blocks;
                    m_i_blocks = (m_i_grid_size + m_i_block_size - 1) / m_i_block_size;

                    assert(m_i_block_size > 0 && m_j_block_size > 0);
                }

                /**
                 * @brief Computes the effective (clamped) block size and position for k-serial stencils.
                 *
                 * @param i_block_index Block index along i-axis.
                 * @param j_block_index Block index along j-axis.
                 *
                 * @return An execution info instance with the computed properties.
                 */
                GT_FORCE_INLINE execinfo_block_kserial block(int_t i_block_index, int_t j_block_index) const {
                    return {i_block_index,
                        j_block_index,
                        clamped_block_size(m_i_grid_size, i_block_index, m_i_block_size, m_i_blocks),
                        clamped_block_size(m_j_grid_size, j_block_index, m_j_block_size, m_j_blocks)};
                }

                /**
                 * @brief Computes the effective (clamped) block size and position for k-parallel stencils.
                 *
                 * @param i_block_index Block index along i-axis.
                 * @param j_block_index Block index along j-axis.
                 * @param k Index along k-axis.
                 *
                 * @return An execution info instance with the computed properties.
                 */
                GT_FORCE_INLINE execinfo_block_kparallel block(
                    int_t i_block_index, int_t j_block_index, int_t k) const {
                    return {i_block_index,
                        j_block_index,
                        k,
                        clamped_block_size(m_i_grid_size, i_block_index, m_i_block_size, m_i_blocks),
                        clamped_block_size(m_j_grid_size, j_block_index, m_j_block_size, m_j_blocks)};
                }

                /** @brief Number of blocks along i-axis. */
                GT_FORCE_INLINE int_t i_blocks() const { return m_i_blocks; }
                /** @brief Number of blocks along j-axis. */
                GT_FORCE_INLINE int_t j_blocks() const { return m_j_blocks; }

                /** @brief Unclamped block size along i-axis. */
                GT_FORCE_INLINE int_t i_block_size() const { return m_i_block_size; }
                /** @brief Unclamped block size along j-axis. */
                GT_FORCE_INLINE int_t j_block_size() const { return m_j_block_size; }
            };
        } // namespace cpu_ifirst_backend
    }     // namespace stencil
} // namespace gridtools
