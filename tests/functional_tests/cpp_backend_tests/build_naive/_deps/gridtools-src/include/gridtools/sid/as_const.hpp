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

#include "../common/host_device.hpp"
#include "../meta/macros.hpp"
#include "concept.hpp"
#include "delegate.hpp"
#include "simple_ptr_holder.hpp"

namespace gridtools {
    namespace sid {
        namespace as_const_impl_ {
            template <class Sid>
            struct const_adapter : delegate<Sid> {
                struct const_ptr_holder {
                    ptr_holder_type<Sid> m_impl;

                    constexpr GT_FUNCTION std::add_const_t<sid::element_type<Sid>> *operator()() const {
                        return m_impl();
                    }

                    friend constexpr GT_FUNCTION const_ptr_holder operator+(
                        const_ptr_holder const &obj, ptr_diff_type<Sid> offset) {
                        return {obj.m_impl + offset};
                    }
                };

                friend const_ptr_holder sid_get_origin(const_adapter const &obj) {
                    return {get_origin(const_cast<Sid &>(obj.m_impl))};
                }
                using delegate<Sid>::delegate;
            };
        } // namespace as_const_impl_

        /**
         *   Returns a `SID`, which ptr_type is a pointer to const.
         *   If the original ptr_type is not a non const pointer `as_const` returns the argument.
         *
         *   TODO(anstaf): at a moment the generated ptr holder always has `host_device` `operator()`
         *                 probably might we need the `host` and `device` variations as well
         */
        template <class Src,
            class Ptr = sid::ptr_type<std::decay_t<Src>>,
            std::enable_if_t<std::is_pointer_v<Ptr> && !std::is_const_v<std::remove_pointer_t<Ptr>>, int> = 0>
        as_const_impl_::const_adapter<Src> as_const(Src &&src) {
            return {std::forward<Src>(src)};
        }

        template <class Src,
            class Ptr = sid::ptr_type<std::decay_t<Src>>,
            std::enable_if_t<!std::is_pointer_v<Ptr> || std::is_const_v<std::remove_pointer_t<Ptr>>, int> = 0>
        decltype(auto) as_const(Src &&src) {
            return std::forward<Src>(src);
        }

        template <class Src>
        decltype(auto) add_const(std::false_type, Src &&src) {
            return std::forward<Src>(src);
        }

        template <class Src>
        decltype(auto) add_const(std::true_type, Src &&src) {
            return sid::as_const(std::forward<Src>(src));
        }
    } // namespace sid
} // namespace gridtools
