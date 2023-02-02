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
#include <cstddef>
#include <type_traits>
#include <utility>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"

namespace gridtools {
    namespace storage {
        namespace info_impl_ {

            template <class>
            struct layout_tuple;

            template <int... Is>
            struct layout_tuple<layout_map<Is...>> {
                using type = tuple<integral_constant<int, Is>...>;
            };

            template <class Layout, class Align>
            struct make_padded_length_f {
                Align m_align;
                template <class Length>
                integral_constant<int, 1> operator()(integral_constant<int, -1>, Length) const {
                    return {};
                }
                template <int Max = Layout::max_arg, class Length, std::enable_if_t<Max != 0 && Max != -1, int> = 0>
                auto operator()(integral_constant<int, Layout::max_arg>, Length length) const {
                    return (length + m_align - integral_constant<int, 1>()) / m_align * m_align;
                }
                template <int I, class Length>
                auto operator()(integral_constant<int, I>, Length length) const {
                    return length;
                }
            };

            template <class Layout, class Align, class Lengths>
            auto make_padded_lengths(Align align, Lengths const &lengths) {
                return tuple_util::transform(
                    make_padded_length_f<Layout, Align>{align}, typename layout_tuple<Layout>::type(), lengths);
            }

            template <class Layout, class Lengths>
            struct make_stride_f {
                Lengths const &m_lengths;
                integral_constant<int, 0> operator()(integral_constant<int, -1>) const { return {}; }
                template <int Max = Layout::max_arg, std::enable_if_t<Max != -1, int> = 0>
                integral_constant<int, 1> operator()(integral_constant<int, Layout::max_arg>) const {
                    return {};
                }
                template <int I>
                auto operator()(integral_constant<int, I>) const {
                    static constexpr size_t next = Layout::find(I + 1);
                    return tuple_util::get<next>(m_lengths) * (*this)(integral_constant<int, I + 1>());
                }
            };

            template <class Layout, class Lengths>
            auto make_strides_helper(Lengths const &lengths) {
                return tuple_util::transform(
                    make_stride_f<Layout, Lengths>{lengths}, typename layout_tuple<Layout>::type());
            }

            template <class Layout, class Align, class Lengths>
            auto make_strides(Align align, Lengths const &lengths) {
                return make_strides_helper<Layout>(make_padded_lengths<Layout>(align, lengths));
            }

            template <class Lengths, class Strides, class = std::make_index_sequence<tuple_util::size<Lengths>::value>>
            class info;

            template <class Lengths, class Strides, size_t... Dims>
            class info<Lengths, Strides, std::index_sequence<Dims...>> : tuple<Lengths, Strides> {
                static_assert(tuple_util::size<Lengths>::value == tuple_util::size<Strides>::value, GT_INTERNAL_ERROR);

                GT_FUNCTION tuple<Lengths, Strides> const &base() const { return *this; }

              public:
                static constexpr size_t ndims = tuple_util::size<Lengths>::value;

                info(Lengths lengths, Strides strides)
                    : tuple<Lengths, Strides>{std::move(lengths), std::move(strides)} {}

                GT_FUNCTION auto const &native_lengths() const { return tuple_util::host_device::get<0>(base()); }
                GT_FUNCTION auto const &native_strides() const { return tuple_util::host_device::get<1>(base()); }
                GT_FUNCTION int length() const {
                    using tuple_util::host_device::get;
                    bool is_empty = false;
                    for (auto empty_dim : {(get<Dims>(native_lengths()) == 0)...})
                        is_empty = is_empty || empty_dim;
                    return is_empty ? 0 : index((get<Dims>(native_lengths()) - integral_constant<int_t, 1>())...) + 1;
                }
                GT_FUNCTION array<uint_t, ndims> lengths() const {
                    return {(uint_t)tuple_util::host_device::get<Dims>(native_lengths())...};
                }
                GT_FUNCTION array<uint_t, ndims> strides() const {
                    return {(uint_t)tuple_util::host_device::get<Dims>(native_strides())...};
                }

                template <class... Is,
                    std::enable_if_t<sizeof...(Is) == ndims &&
                                         std::conjunction<std::is_convertible<Is, int_t>...>::value,
                        int> = 0>
                GT_FUNCTION auto index(Is... indices) const {
                    return index_from_tuple(tuple(indices...));
                }

                template <class Indices>
                GT_FUNCTION auto index_from_tuple(Indices &&indices) const {
                    using namespace tuple_util::host_device;
#ifndef NDEBUG
                    tuple_util::host_device::for_each(
                        [](int index, int length) {
                            assert(index >= 0);
                            assert(index < length);
                        },
                        indices,
                        native_lengths());
#endif
                    return fold([](auto l, auto r) { return l + r; },
                        transform(
                            [](auto i, auto s) { return i * s; }, std::forward<Indices>(indices), native_strides()));
                }
            };

            template <class Lengths, class Strides>
            info<Lengths, Strides> make_info_helper(Lengths lengths, Strides strides) {
                return {std::move(lengths), std::move(strides)};
            }

            template <class Layout, class Align, class Lengths>
            auto make_info(Align align, Lengths const &lengths) {
                return make_info_helper(lengths, make_strides<Layout>(align, lengths));
            }
        } // namespace info_impl_
        using info_impl_::make_info;
    } // namespace storage
} // namespace gridtools
