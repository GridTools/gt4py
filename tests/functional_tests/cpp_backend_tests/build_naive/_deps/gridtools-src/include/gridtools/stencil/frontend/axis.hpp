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

#include <algorithm>
#include <type_traits>
#include <utility>

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../core/interval.hpp"
#include "../core/level.hpp"

namespace gridtools {
    namespace stencil {
        namespace axis_config {
            template <int_t V>
            struct offset_limit : std::integral_constant<int_t, V> {};
        } // namespace axis_config

        /**
         * Defines an axis_interval_t which spans the whole axis.
         * @param NIntervals Number of intervals the axis should support
         * @param LevelOffsetLimit Maximum offset relative to the splitter position that is required to specify the
         * intervals
         */
        template <size_t, class = axis_config::offset_limit<2>>
        class axis;

        template <size_t NIntervals, int_t LevelOffsetLimit>
        class axis<NIntervals, axis_config::offset_limit<LevelOffsetLimit>> {
            template <size_t Id, size_t... Ids>
            struct interval_impl {
                static_assert(
                    std::is_same_v<std::make_index_sequence<sizeof...(Ids) + 1>, std::index_sequence<0, Ids - Id...>>,
                    "Intervals must be continuous.");
                static_assert(Id + sizeof...(Ids) < NIntervals, "Interval ID out of bounds for this axis.");
                using type = core::interval<core::level<Id, 1, LevelOffsetLimit>,
                    core::level<Id + sizeof...(Ids) + 1, -1, LevelOffsetLimit>>;
            };

          public:
            static constexpr size_t n_intervals = NIntervals;

            using axis_interval_t =
                core::interval<core::level<0, 1, LevelOffsetLimit>, core::level<NIntervals, -1, LevelOffsetLimit>>;

            using full_interval =
                core::interval<core::level<0, 1, LevelOffsetLimit>, core::level<NIntervals, -1, LevelOffsetLimit>>;

            template <typename... IntervalSizes,
                std::enable_if_t<sizeof...(IntervalSizes) == NIntervals &&
                                     std::conjunction<std::is_convertible<IntervalSizes, int_t>...>::value,
                    int> = 0>
            axis(IntervalSizes... interval_sizes) : interval_sizes_{static_cast<int_t>(interval_sizes)...} {}

            int_t interval_size(size_t index) const { return interval_sizes_[index]; }
            const array<int_t, NIntervals> &interval_sizes() const { return interval_sizes_; };

            template <size_t... IntervalIDs>
            using get_interval = typename interval_impl<IntervalIDs...>::type;

          private:
            array<int_t, NIntervals> interval_sizes_;
        };
    } // namespace stencil
} // namespace gridtools
