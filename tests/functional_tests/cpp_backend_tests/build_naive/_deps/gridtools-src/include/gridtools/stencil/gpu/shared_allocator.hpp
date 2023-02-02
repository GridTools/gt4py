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

#include <cstdlib>

#include "../../common/cuda_runtime.hpp"
#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"

namespace gridtools {
    namespace stencil {
        namespace gpu_backend {
            class shared_allocator {
                size_t m_offset = 0; // in bytes

              public:
                template <class T>
                struct lazy_alloc {
                    size_t m_offset;

                    GT_FUNCTION_DEVICE T *operator()() const {
                        extern __shared__ char ij_cache_shm[];
                        return reinterpret_cast<T *>(ij_cache_shm + m_offset);
                    }

                    friend GT_FUNCTION lazy_alloc operator+(lazy_alloc l, ptrdiff_t r) {
                        l.m_offset += r * sizeof(T);
                        return l;
                    }
                };

                /**
                 * \param size size of allocation in number of elements
                 */
                template <class LazyT>
                friend auto allocate(shared_allocator &obj, LazyT, size_t size) {
                    using type = typename LazyT::type;
                    static constexpr auto alignment = alignof(type);
                    auto aligned = (obj.m_offset + alignment - 1) / alignment * alignment;
                    obj.m_offset = aligned + size * sizeof(type);
                    return lazy_alloc<type>{aligned};
                }

                size_t size() const { return m_offset; }
            };
        } // namespace gpu_backend
    }     // namespace stencil
} // namespace gridtools
