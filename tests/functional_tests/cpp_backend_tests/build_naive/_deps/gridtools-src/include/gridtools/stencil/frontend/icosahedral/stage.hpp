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

#include "../../../common/defs.hpp"
#include "../../../common/for_each.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/hymap.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../../meta.hpp"
#include "../../../sid/composite.hpp"
#include "../../../sid/concept.hpp"
#include "../../../sid/multi_shift.hpp"
#include "../../common/dim.hpp"
#include "../../common/extent.hpp"
#include "../../common/intent.hpp"
#include "connectivity.hpp"
#include "location_type.hpp"

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *   Stage concept for icosahedral grid is defined similar as for structured grid (with some additions)
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation area by
 *   that much.
 *
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor for all colors from the grid point that IteratorDomain points to.
 *   precondition: IteratorDomain should point to the first color.
 *   postcondition: IteratorDomain still points to the first color.
 *
 *   Stage has templated variation of `exec` which accept color number as a first template parameter. This variation
 *   does not iterate on colors; it executes an elementary functor for the given color.
 *   precondition: IteratorDomain should point to the same color as one in exec parameter.
 *
 *   Stage has netsted metafunction contains_color<Color> that evaluates to std::false_type if for the given color
 *   the elementary function is not executed.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

namespace gridtools {
    namespace stencil {
        namespace icosahedral {
            namespace stage_impl_ {
                struct default_deref_f {
                    template <class Key, class T>
                    GT_FUNCTION decltype(auto) operator()(Key, T ptr) const {
                        return *ptr;
                    }
                };

                template <class Ptr, class Strides, class Keys, class Deref, class LocationType, int_t Color>
                struct evaluator {
                    Ptr const &m_ptr;
                    Strides const &m_strides;

                    template <class Key, class Offset>
                    GT_FUNCTION decltype(auto) get_ref(Offset offset) const {
                        return Deref()(Key(),
                            sid::multi_shifted<Key>(host_device::at_key<Key>(m_ptr), m_strides, std::move(offset)));
                    }

                    template <class Accessor>
                    GT_FUNCTION decltype(auto) operator()(Accessor) const {
                        return apply_intent<Accessor::intent_v>(get_ref<meta::at_c<Keys, Accessor::index_t::value>>(
                            hymap::keys<dim::c>::values<integral_constant<int_t, Color>>()));
                    }

                    template <class Accessor, class Offset>
                    GT_FUNCTION decltype(auto) neighbor(Accessor, Offset offset) const {
                        return apply_intent<Accessor::intent_v>(
                            get_ref<meta::at_c<Keys, Accessor::index_t::value>>(std::move(offset)));
                    }

                    static constexpr int_t color = Color;

                    template <class Fun, class Accessor, class... Accessors>
                    GT_FUNCTION void for_neighbors(Fun &&fun, Accessor, Accessors...) const {
                        static_assert(
                            std::conjunction_v<
                                std::is_same<typename Accessor::location_t, typename Accessors::location_t>...>,
                            "All accessors should be of the same location");
                        host_device::for_each<neighbor_offsets<LocationType, typename Accessor::location_t, Color>>(
                            [&](auto offset) { fun(neighbor(Accessor(), offset), neighbor(Accessors(), offset)...); });
                    }
                };

                template <class Functor, class PlhMap>
                struct stage {
                    using location_t = typename Functor::location;

                    template <class Deref = void, class Ptr, class Strides>
                    GT_FUNCTION void operator()(Ptr const &ptr, Strides const &strides) const {
                        using deref_t = meta::if_<std::is_void<Deref>, default_deref_f, Deref>;
                        host_device::for_each<meta::make_indices<location_t>>([&](auto color) {
                            using eval_t = evaluator<Ptr, Strides, PlhMap, deref_t, location_t, decltype(color)::value>;
                            Functor::apply(eval_t{ptr, strides});
                        });
                    }
                };
            } // namespace stage_impl_
            template <class... Ts>
            meta::curry<stage_impl_::stage> get_stage(Ts &&...);
        } // namespace icosahedral
    }     // namespace stencil
} // namespace gridtools
