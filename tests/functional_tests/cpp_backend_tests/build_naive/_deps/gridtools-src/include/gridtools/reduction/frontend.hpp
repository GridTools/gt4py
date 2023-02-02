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

#include <cassert>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>

#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/allocator.hpp"
#include "../storage/traits.hpp"

namespace gridtools {
    namespace reduction {
        namespace frontend_impl_ {
            template <class Sizes>
            inline auto zeros(Sizes const &sizes) {
                return tuple_util::transform([](auto &&) { return integral_constant<int_t, 0>(); }, sizes);
            }

            template <class Sizes>
            using zeros_type = decltype(zeros(std::declval<Sizes const &>()));

            template <class Backend, class T, class Origin, class Strides, class StridesKind, class Sizes>
            struct reducible {
                std::shared_ptr<void> m_alloc;
                T neutral_value;
                Origin m_origin;
                size_t m_size;
                Strides m_strides;
                Sizes m_sizes;

                template <class F>
                auto reduce(F f) const {
                    assert(m_size);
                    return reduction_reduce(Backend(), neutral_value, f, m_origin(), m_size);
                }

                friend Strides sid_get_strides(reducible const &obj) { return obj.m_strides; }
                friend Origin sid_get_origin(reducible const &obj) { return {obj.m_origin}; }
                friend zeros_type<Sizes> sid_get_lower_bounds(reducible const &obj) { return zeros(obj.m_sizes); }
                friend Sizes sid_get_upper_bounds(reducible const &obj) { return obj.m_sizes; }
            };

            template <class Backend, class T, class Origin, class Strides, class StridesKind, class Sizes>
            StridesKind sid_get_strides_kind(reducible<Backend, T, Origin, Strides, StridesKind, Sizes> const &);

            template <class StorageTraits>
            struct alloc_fun {
                auto operator()(size_t size) const { return storage::traits::allocate<StorageTraits, char>(size); }
            };

            template <class Backend, class StorageTraits, class Id = void, class T, class... Dims>
            auto make_reducible(T const &neutral_value, Dims... dims) {
                sid::host_device::cached_allocator<alloc_fun<StorageTraits>> alloc;
                auto lengths = tuple(dims...);
                auto info = storage::traits::make_info<StorageTraits, T>(lengths);
                auto strides = info.native_strides();
                size_t data_size = info.length();
                size_t rounded_size = reduction_round_size(Backend(), data_size);
                size_t allocation_size = reduction_allocation_size(Backend(), rounded_size);
                auto origin = allocate(alloc, meta::lazy::id<T>(), allocation_size);
                reduction_fill(Backend(),
                    neutral_value,
                    origin(),
                    data_size,
                    rounded_size,
                    storage::traits::has_holes<StorageTraits, T>(lengths));
                return reducible<Backend,
                    T,
                    decltype(origin),
                    decltype(strides),
                    storage::traits::strides_kind<StorageTraits, T, decltype(lengths), Id>,
                    decltype(lengths)>{std::make_shared<decltype(alloc)>(std::move(alloc)),
                    neutral_value,
                    std::move(origin),
                    rounded_size,
                    std::move(strides),
                    std::move(lengths)};
            }
        } // namespace frontend_impl_
        using frontend_impl_::make_reducible;
    } // namespace reduction
} // namespace gridtools
