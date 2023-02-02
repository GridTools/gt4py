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

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "level.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace interval_impl {
                constexpr int_t sign(int_t value) { return (0 < value) - (value < 0); }
                constexpr int_t add_offset(int_t offset, int_t value) {
                    return sign(offset + value) == sign(offset) ? offset + value : offset + value + sign(value);
                }

                /**
                 * @struct Interval
                 * Structure defining a closed interval on an axis given two levels
                 */
                template <class, class>
                struct interval;

                template <uint_t FromSplitter, int_t FromOffset, uint_t ToSplitter, int_t ToOffset, int_t OffsetLimit>
                struct interval<level<FromSplitter, FromOffset, OffsetLimit>,
                    level<ToSplitter, ToOffset, OffsetLimit>> {
                    // check the from level is lower or equal to the to level
                    static_assert(FromSplitter < ToSplitter || (FromSplitter == ToSplitter && FromOffset <= ToOffset),
                        "check the from level is lower or equal to the to level");
                    static constexpr int_t offset_limit = OffsetLimit;

                    // User API: helper to access the first and last level as an interval
                    using first_level = interval<level<FromSplitter, FromOffset, OffsetLimit>,
                        level<FromSplitter, FromOffset, OffsetLimit>>;
                    using last_level =
                        interval<level<ToSplitter, ToOffset, OffsetLimit>, level<ToSplitter, ToOffset, OffsetLimit>>;

                    /**
                     * @brief returns an interval where the boundaries are modified according to left and right
                     * @param left moves the left boundary, the interval is enlarged (left < 0) or shrunk (left > 0)
                     * @param right moves the right boundary, the interval is enlarged (right > 0) or shrunk (right < 0)
                     */
                    template <int_t left, int_t right>
                    struct modify_impl {
                        static_assert(
                            add_offset(FromOffset, left) >= -OffsetLimit && add_offset(ToOffset, right) <= OffsetLimit,
                            "You are trying to modify an interval to increase beyond its maximal offset.");
                        static_assert(
                            FromSplitter < ToSplitter || add_offset(FromOffset, left) <= add_offset(ToOffset, right),
                            "You are trying to modify an interval such that the result is an empty interval(left "
                            "boundary "
                            "> right boundary).");
                        using type = interval<level<FromSplitter, add_offset(FromOffset, left), OffsetLimit>,
                            level<ToSplitter, add_offset(ToOffset, right), OffsetLimit>>;
                    };
                    template <int_t left, int_t right>
                    using modify = typename modify_impl<left, right>::type;
                    template <int_t dir>
                    using shift = modify<dir, dir>;
                };

                template <class, class>
                struct concat_folder;

                template <class From, class Level, class NextLevel, class To>
                struct concat_folder<interval<From, Level>, interval<NextLevel, To>> {
                    static_assert(
                        level_to_index<Level>::value + 1 == level_to_index<NextLevel>::value, GT_INTERNAL_ERROR);
                    using type = interval<From, To>;
                };

                template <class...>
                struct concat_intervals;

                template <class T>
                struct concat_intervals<T> {
                    using type = T;
                };

                template <class T, class... Ts>
                struct concat_intervals<T, Ts...> {
                    using type = meta::foldl<meta::force<concat_folder>::apply, T, meta::list<Ts...>>;
                };

                template <class...>
                struct enclosing_folder;

                template <class LFrom, class LTo, class RFrom, class RTo>
                struct enclosing_folder<interval<LFrom, LTo>, interval<RFrom, RTo>> {
                    using from_t =
                        meta::if_c<(level_to_index<LFrom>::value < level_to_index<RFrom>::value), LFrom, RFrom>;
                    using to_t = meta::if_c<(level_to_index<LTo>::value > level_to_index<RTo>::value), LTo, RTo>;
                    using type = interval<from_t, to_t>;
                };

                template <class...>
                struct enclosing_interval;

                template <class T>
                struct enclosing_interval<T> {
                    using type = T;
                };

                template <class T, class... Ts>
                struct enclosing_interval<T, Ts...> {
                    using type = meta::foldl<meta::force<enclosing_folder>::apply, T, meta::list<Ts...>>;
                };
            } // namespace interval_impl
            using interval_impl::interval;
            template <class... Ts>
            using concat_intervals = typename interval_impl::concat_intervals<Ts...>::type;
            template <class... Ts>
            using enclosing_interval = typename interval_impl::enclosing_interval<Ts...>::type;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
