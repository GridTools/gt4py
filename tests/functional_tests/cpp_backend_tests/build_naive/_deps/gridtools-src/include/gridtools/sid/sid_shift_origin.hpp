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

#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "concept.hpp"
#include "delegate.hpp"
#include "multi_shift.hpp"

namespace gridtools {
    namespace sid {
        namespace shift_sid_origin_impl_ {

            template <class Offsets>
            struct add_offset_f {
                Offsets const &m_offsets;

                template <class Dim, class Bound, std::enable_if_t<has_key<Offsets, Dim>::value, int> = 0>
                auto operator()(Bound &&bound) const {
                    return std::forward<Bound>(bound) - at_key<Dim>(m_offsets);
                }

                template <class Dim, class Bound, std::enable_if_t<!has_key<Offsets, Dim>::value, int> = 0>
                std::decay_t<Bound> operator()(Bound &&bound) const {
                    return bound;
                }
            };

            template <class Bounds, class Offsets>
            auto add_offsets(Bounds &&bounds, Offsets const &offsets) {
                return hymap::transform(add_offset_f<Offsets>{offsets}, std::forward<Bounds>(bounds));
            }

            template <class Sid, class LowerBounds, class UpperBounds>
            class shifted_sid : public delegate<Sid> {
                ptr_holder_type<Sid> m_origin;
                LowerBounds m_lower_bounds;
                UpperBounds m_upper_bounds;

                friend ptr_holder_type<Sid> sid_get_origin(shifted_sid const &obj) { return obj.m_origin; }
                friend LowerBounds const &sid_get_lower_bounds(shifted_sid const &obj) { return obj.m_lower_bounds; }
                friend UpperBounds const &sid_get_upper_bounds(shifted_sid const &obj) { return obj.m_upper_bounds; }

              public:
                template <class Arg, class Offsets>
                shifted_sid(Arg &&original_sid, Offsets &&offsets) noexcept
                    : delegate<Sid>(std::forward<Arg>(original_sid)),
                      m_origin(get_origin(this->m_impl) +
                               multi_shifted(ptr_diff_type<Sid>(), get_strides(this->m_impl), offsets)),
                      m_lower_bounds(add_offsets(get_lower_bounds(this->m_impl), offsets)),
                      m_upper_bounds(add_offsets(get_upper_bounds(this->m_impl), offsets)) {}
            };

            template <class Sid, class Offsets>
            using shifted_sid_type = shifted_sid<Sid,
                decltype(add_offsets(get_lower_bounds(std::declval<Sid const &>()), std::declval<Offsets>())),
                decltype(add_offsets(get_upper_bounds(std::declval<Sid const &>()), std::declval<Offsets>()))>;
        } // namespace shift_sid_origin_impl_

        template <class Sid, class Offset>
        shift_sid_origin_impl_::shifted_sid_type<Sid, Offset> shift_sid_origin(Sid &&sid, Offset &&offset) {
            return {std::forward<Sid>(sid), std::forward<Offset>(offset)};
        }
    } // namespace sid
} // namespace gridtools
