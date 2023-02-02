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

#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"

namespace gridtools {
    namespace storage {
        namespace cpu_kfirst_impl_ {
            /**
             * @brief metafunction used to retrieve a layout_map with n-dimensions that can be used in combination with
             * the host backend (k-first order). E.g., make_layout<5> will return following type: layout_map<2,3,4,0,1>.
             * This means the k-dimension (value: 4) is coalesced in memory, followed by the j-dimension (value: 3),
             * followed by the i-dimension (value: 2), followed by the fifth dimension (value: 1), etc. The reason for
             * having k as innermost is because of the gridtools execution model. The CPU backend will give best
             * performance (in most cases) when using the provided layout.
             */
            template <size_t N, class = std::make_index_sequence<N>>
            struct make_layout;

            template <size_t N, size_t... Dims>
            struct make_layout<N, std::index_sequence<Dims...>> {
                using type = layout_map<Dims...>;
            };

            template <size_t N, size_t Dim0, size_t Dim1, size_t Dim2, size_t... Dims>
            struct make_layout<N, std::index_sequence<Dim0, Dim1, Dim2, Dims...>> {
                using type = layout_map<Dim0 + N - 3, Dim1 + N - 3, Dim2 + N - 3, (Dims - 3)...>;
            };
        } // namespace cpu_kfirst_impl_

        struct cpu_kfirst {
            friend std::true_type storage_is_host_referenceable(cpu_kfirst) { return {}; }

            template <size_t Dims>
            friend typename cpu_kfirst_impl_::make_layout<Dims>::type storage_layout(
                cpu_kfirst, std::integral_constant<size_t, Dims>) {
                return {};
            }

            friend integral_constant<size_t, 1> storage_alignment(cpu_kfirst) { return {}; }

            template <class LazyType, class T = typename LazyType::type>
            friend auto storage_allocate(cpu_kfirst, LazyType, size_t size) {
                return std::make_unique<T[]>(size);
            }
        };
    } // namespace storage
} // namespace gridtools
