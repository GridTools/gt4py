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
#include <type_traits>
#include <utility>

#include "../../../common/array.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/functional.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../../common/tuple.hpp"
#include "../../../meta.hpp"
#include "../../common/extent.hpp"
#include "../../common/intent.hpp"
#include "dimension.hpp"

/**
   @file

   @brief File containing the definition of the regular accessor used
   to address the storage (at offsets) from within the functors.
   This accessor is a proxy for a storage class, i.e. it is a light
   object used in place of the storage when defining the high level
   computations, and it will be bound later on with a specific
   instantiation of a storage class.

   An accessor can be instantiated directly in the apply
   method, or it might be a constant expression instantiated outside
   the functor scope and with static duration.
*/

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace accessor_impl_ {
                template <size_t I>
                GT_FUNCTION constexpr int_t pick_dimension() {
                    return 0;
                }

                template <size_t I, class... Ts>
                GT_FUNCTION constexpr int_t pick_dimension(dimension<I> src, Ts &&...) {
                    return src.value;
                }

                template <size_t I, size_t J, class... Ts, std::enable_if_t<I != J, int> = 0>
                GT_FUNCTION constexpr int_t pick_dimension(dimension<J>, Ts... srcs) {
                    return pick_dimension<I>(srcs...);
                }

                template <size_t>
                struct just_int {
                    using type = int_t;
                };

                template <class... Ts>
                using are_ints = std::conjunction<std::is_convertible<Ts, int_t>...>;

                template <size_t Dim, size_t I, std::enable_if_t<(I > Dim), int> = 0>
                GT_FUNCTION constexpr int_t out_of_range_dim(dimension<I> obj) {
                    return obj.value;
                }

                template <size_t Dim, size_t I, std::enable_if_t<(I <= Dim), int> = 0>
                GT_FUNCTION constexpr int_t out_of_range_dim(dimension<I>) {
                    return 0;
                }

                template <class Extent>
                struct minimal_dim : std::integral_constant<size_t, 3> {};

                template <>
                struct minimal_dim<extent<>> : std::integral_constant<size_t, 0> {};

                template <int_t IMinus, int_t IPlus>
                struct minimal_dim<extent<IMinus, IPlus>> : std::integral_constant<size_t, 1> {};

                template <int_t IMinus, int_t IPlus, int_t JMinus, int_t JPlus>
                struct minimal_dim<extent<IMinus, IPlus, JMinus, JPlus>> : std::integral_constant<size_t, 2> {};

                template <class>
                struct minimal_requried_args;

                template <int_t IMinus, int_t IPlus, int_t JMinus, int_t JPlus, int_t KMinus, int_t KPlus>
                struct minimal_requried_args<extent<IMinus, IPlus, JMinus, JPlus, KMinus, KPlus>>
                    : std::integral_constant<size_t,
                          (KMinus > 0 || KPlus < 0      ? 3
                              : JMinus > 0 || JPlus < 0 ? 2
                              : IMinus > 0 || IPlus < 0 ? 1
                                                        : 0)> {};

#ifndef NDEBUG
                template <class Extent, size_t Dim>
                GT_FUNCTION constexpr bool check_offsets(array<int_t, Dim> const &offsets) {
                    int_t i = Dim > 0 ? offsets[0] : 0;
                    int_t j = Dim > 1 ? offsets[1] : 0;
                    int_t k = Dim > 2 ? offsets[2] : 0;
                    return i >= Extent::iminus::value && i <= Extent::iplus::value && j >= Extent::jminus::value &&
                           j <= Extent::jplus::value && k >= Extent::kminus::value && k <= Extent::kplus::value;
                }
#endif
            } // namespace accessor_impl_

            template <uint_t Id,
                intent Intent = intent::in,
                class Extent = extent<>,
                size_t Dim = accessor_impl_::minimal_dim<Extent>::value,
                class = std::make_index_sequence<Dim>>
            class accessor;

            template <uint_t Id, intent Intent, class Extent, size_t Dim, size_t... Is>
            class accessor<Id, Intent, Extent, Dim, std::index_sequence<Is...>> : public array<int_t, Dim> {
                using base_t = array<int_t, Dim>;

              public:
                using index_t = integral_constant<uint_t, Id>;
                static constexpr intent intent_v = Intent;
                using extent_t = Extent;

                template <class... Ts,
                    std::enable_if_t<sizeof...(Ts) < Dim && std::conjunction<std::is_convertible<Ts, int_t>...>::value,
                        int> = 0>
                GT_FUNCTION constexpr accessor(Ts... offsets) : base_t({offsets...}) {
                    static_assert(sizeof...(Ts) >= accessor_impl_::minimal_requried_args<Extent>::value,
                        "zero offsets is out of the extents range");
                    assert(accessor_impl_::check_offsets<Extent>(*this));
                }

                template <class... Ts, std::enable_if_t<accessor_impl_::are_ints<Ts...>::value, int> = 0>
                GT_FUNCTION constexpr accessor(typename accessor_impl_::just_int<Is>::type... offsets, Ts... zeros)
                    : base_t({offsets...}) {
                    assert((true && ... && (zeros == 0)));
                    assert(accessor_impl_::check_offsets<Extent>(*this));
                }

                template <size_t J, size_t... Js>
                GT_FUNCTION constexpr accessor(dimension<J> src, dimension<Js>... srcs)
                    : base_t({accessor_impl_::pick_dimension<Is + 1>(src, srcs...)...}) {
                    static_assert(meta::is_set_fast<meta::list<dimension<J>, dimension<Js>...>>::value,
                        "all dimensions should be of different indicies");
                    assert(accessor_impl_::out_of_range_dim<Dim>(src) == 0);
                    assert((true && ... && (accessor_impl_::out_of_range_dim<Dim>(srcs) == 0)));
                    assert(accessor_impl_::check_offsets<Extent>(*this));
                }
            };

            template <uint_t Id, class Extent, intent Intent>
            class accessor<Id, Intent, Extent, 0, std::index_sequence<>> : public tuple<> {
              public:
                static_assert(std::is_same_v<Extent, extent<>>, GT_INTERNAL_ERROR);

                using index_t = integral_constant<uint_t, Id>;
                static constexpr intent intent_v = Intent;
                using extent_t = Extent;

                accessor() = default;

                template <class... Ts, std::enable_if_t<accessor_impl_::are_ints<Ts...>::value, int> = 0>
                GT_FUNCTION constexpr accessor(Ts... zeros) {
                    assert((true && ... && (zeros == 0)));
                }

                template <size_t J, size_t... Js>
                GT_FUNCTION constexpr accessor(dimension<J> zero, dimension<Js>... zeros) {
                    assert(zero.value == 0);
                    assert((true && ... && (zeros.value == 0)));
                }
            };

            template <uint_t ID, intent Intent, typename Extent, size_t Number>
            meta::repeat_c<Number, meta::list<int_t>> tuple_to_types(accessor<ID, Intent, Extent, Number> const &);

            template <uint_t ID, intent Intent, typename Extent, size_t Number>
            meta::always<accessor<ID, Intent, Extent, Number>> tuple_from_types(
                accessor<ID, Intent, Extent, Number> const &);

            template <class>
            struct is_accessor : std::false_type {};

            template <uint_t ID, intent Intent, typename Extent, size_t Number, class Seq>
            struct is_accessor<accessor<ID, Intent, Extent, Number, Seq>> : std::true_type {};

            template <uint_t ID, typename Extent = extent<>, size_t Number = accessor_impl_::minimal_dim<Extent>::value>
            using in_accessor = accessor<ID, intent::in, Extent, Number>;

            template <uint_t ID, typename Extent = extent<>, size_t Number = accessor_impl_::minimal_dim<Extent>::value>
            using inout_accessor = accessor<ID, intent::inout, Extent, Number>;
        } // namespace cartesian
    }     // namespace stencil
} // namespace gridtools
