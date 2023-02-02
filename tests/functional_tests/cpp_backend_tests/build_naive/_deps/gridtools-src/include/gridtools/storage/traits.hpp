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

#include <numeric>
#include <type_traits>

#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/unknown_kind.hpp"
#include "data_view.hpp"
#include "info.hpp"

namespace gridtools {
    namespace storage {
        namespace traits {
            template <class Traits>
            constexpr bool is_host_referenceable =
                decltype(storage_is_host_referenceable(std::declval<Traits>()))::value;

            template <class Traits>
            constexpr size_t byte_alignment = decltype(storage_alignment(std::declval<Traits>()))::value;

            template <class Traits, class T, size_t ByteAlignment = byte_alignment<Traits>>
            constexpr size_t elem_alignment = ByteAlignment / std::gcd(sizeof(T), ByteAlignment);

            template <class Traits, size_t Dims>
            using layout_type =
                decltype(storage_layout(std::declval<Traits>(), std::integral_constant<size_t, Dims>()));

            template <class Traits, class T, class Lengths>
            auto make_info(Lengths const &lengths) {
                return storage::make_info<layout_type<Traits, tuple_util::size<Lengths>::value>>(
                    integral_constant<int, elem_alignment<Traits, T>>(), lengths);
            }

            template <class Traits,
                class T,
                class Lengths,
                class Id,
                class Info = decltype(make_info<Traits, T, Lengths>(std::declval<Lengths const &>())),
                class Strides = decltype(std::declval<Info const &>().native_strides()),
                class Layout = layout_type<Traits, tuple_util::size<Lengths>::value>>
            using strides_kind = meta::if_<std::is_same<Id, sid::unknown_kind>,
                Id,
                meta::if_<tuple_util::is_empty_or_tuple_of_empties<Strides>,
                    Strides,
                    meta::list<Strides,
                        Layout,
                        Id,
                        meta::if_c<(Layout::unmasked_length > 1),
                            integral_constant<int, elem_alignment<Traits, T>>,
                            void>>>>;

            template <class Traits,
                class T,
                class Lengths,
                size_t Alignment = elem_alignment<Traits, T>,
                std::enable_if_t<Alignment == 1, int> = 0>
            std::false_type has_holes(Lengths const &) {
                return {};
            }

            template <class Traits,
                class T,
                class Lengths,
                size_t Alignment = elem_alignment<Traits, T>,
                size_t Dims = tuple_util::size<Lengths>::value,
                class Layout = layout_type<Traits, Dims>,
                std::enable_if_t<Alignment != 1, int> = 0>
            bool has_holes(Lengths const &lengths) {
                return tuple_util::get<Layout::find(Dims - 1)>(lengths) % Alignment;
            }

            template <class Traits, class T>
            auto allocate(size_t size) {
                return storage_allocate(Traits(), meta::lazy::id<T>(), size);
            }

            template <class Traits, class T>
            using target_ptr_type = decltype(allocate<Traits, T>(0));

            template <class Traits, class T>
            std::enable_if_t<!is_host_referenceable<Traits>> update_target(T *dst, T const *src, size_t size) {
                storage_update_target(Traits(), dst, src, size);
            }

            template <class Traits, class T>
            std::enable_if_t<!is_host_referenceable<Traits>> update_host(T *dst, T const *src, size_t size) {
                storage_update_host(Traits(), dst, src, size);
            }

            template <class Traits, class T, class Info, std::enable_if_t<is_host_referenceable<Traits>, int> = 0>
            auto storage_make_target_view(Traits, T *ptr, Info const &info) {
                return make_host_view(ptr, info);
            }

            template <class Traits, class T, class Info>
            auto make_target_view(T *ptr, Info const &info) {
                return storage_make_target_view(Traits(), ptr, info);
            }
        } // namespace traits
    }     // namespace storage
} // namespace gridtools
