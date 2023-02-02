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

#include <memory>
#include <type_traits>

#include "../common/hugepage_alloc.hpp"
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"

namespace gridtools {
    namespace storage {
        namespace cpu_ifirst_impl_ {
            template <size_t N, class = std::make_index_sequence<N>>
            struct make_layout;

            template <size_t N, size_t... Dims>
            struct make_layout<N, std::index_sequence<Dims...>> {
                using type = layout_map<(N - 1 - Dims)...>;
            };

            template <size_t N, size_t Dim0, size_t Dim1, size_t Dim2, size_t... Dims>
            struct make_layout<N, std::index_sequence<Dim0, Dim1, Dim2, Dims...>> {
                using type = layout_map<N - 1 - Dim0, N - 1 - Dim2, N - 1 - Dim1, (N - 1 - Dims)...>;
            };

            struct deleter {
                template <class T>
                void operator()(T *p) const {
                    hugepage_free(const_cast<std::remove_cv_t<T> *>(p));
                }
            };
        } // namespace cpu_ifirst_impl_

        struct cpu_ifirst {
            friend std::true_type storage_is_host_referenceable(cpu_ifirst) { return {}; }

            template <size_t Dims>
            friend typename cpu_ifirst_impl_::make_layout<Dims>::type storage_layout(
                cpu_ifirst, std::integral_constant<size_t, Dims>) {
                return {};
            }

            friend integral_constant<size_t, 64> storage_alignment(cpu_ifirst) { return {}; }

            template <class LazyType, class T = typename LazyType::type>
            friend auto storage_allocate(cpu_ifirst, LazyType, size_t size) {
                return std::unique_ptr<T[], cpu_ifirst_impl_::deleter>(
                    static_cast<T *>(hugepage_alloc(size * sizeof(T))));
            }
        };
    } // namespace storage
} // namespace gridtools
