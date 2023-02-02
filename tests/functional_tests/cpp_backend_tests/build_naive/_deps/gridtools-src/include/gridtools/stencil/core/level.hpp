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

#include "../../common/defs.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace level_impl_ {
                constexpr int_t calc_level_index(uint_t splitter, int_t offset, int_t limit) {
                    return limit * (2 * (int_t)splitter + 1) + offset - (offset >= 0);
                }
                constexpr int_t get_splitter_from_index(int_t index, int_t limit) { return index / (2 * limit); }
                constexpr int_t get_offset_from_index(int_t index, int_t limit) {
                    return index % (2 * limit) - limit + (index % (2 * limit) >= limit);
                }
            } // namespace level_impl_

            /**
             * @struct Level
             * Structure defining an axis position relative to a splitter
             */
            template <uint_t Splitter, int_t Offset, int_t OffsetLimit>
            struct level {
                // check offset and splitter value ranges
                // (note that non negative splitter values simplify the index computation)
                static_assert(Splitter >= 0 && Offset != 0,
                    "check offset and splitter value ranges \n(note that non negative splitter values simplify the "
                    "index computation)");
                static_assert(-OffsetLimit <= Offset && Offset <= OffsetLimit,
                    "check offset and splitter value ranges \n(note that non negative splitter values simplify the "
                    "index computation)");

                // define splitter, level offset and offset limit
                static constexpr uint_t splitter = Splitter;
                static constexpr int_t offset = Offset;
                static constexpr int_t offset_limit = OffsetLimit;
            };

            template <int_t Value, int_t OffsetLimit>
            struct level_index {
                static constexpr int_t value = Value;
                static constexpr int_t offset_limit = OffsetLimit;
            };

            template <class Level>
            using level_to_index =
                level_index<level_impl_::calc_level_index(Level::splitter, Level::offset, Level::offset_limit),
                    Level::offset_limit>;

            /**
             * @struct index_to_level
             * Meta function converting a unique index back into a level
             */
            template <class Index>
            using index_to_level = level<level_impl_::get_splitter_from_index(Index::value, Index::offset_limit),
                level_impl_::get_offset_from_index(Index::value, Index::offset_limit),
                Index::offset_limit>;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
