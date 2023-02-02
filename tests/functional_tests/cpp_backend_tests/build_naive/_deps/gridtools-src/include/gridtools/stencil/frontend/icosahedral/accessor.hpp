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

#include "../../../common/defs.hpp"
#include "../../common/extent.hpp"
#include "../../common/intent.hpp"
#include "location_type.hpp"

namespace gridtools {
    namespace stencil {
        namespace icosahedral {
            /**
             * This is the type of the accessors accessed by a stencil functor.
             */
            template <uint_t Id, intent Intent, class LocationType, class Extent = extent<>>
            struct accessor {
                static_assert(is_location_type<LocationType>::value, GT_INTERNAL_ERROR);
                using index_t = integral_constant<uint_t, Id>;
                static constexpr intent intent_v = Intent;
                using extent_t = Extent;
                using location_t = LocationType;
            };

            template <uint_t ID, typename LocationType, typename Extent = extent<>>
            using in_accessor = accessor<ID, intent::in, LocationType, Extent>;

            template <uint_t ID, typename LocationType>
            using inout_accessor = accessor<ID, intent::inout, LocationType, extent<>>;
        } // namespace icosahedral
    }     // namespace stencil
} // namespace gridtools
