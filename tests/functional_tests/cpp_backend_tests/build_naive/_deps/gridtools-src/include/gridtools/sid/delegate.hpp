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

#include "../common/defs.hpp"
#include "../meta/macros.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        /**
         *  A helper class for implementing delegate design pattern for `SID`s
         *  Typically the user template class should inherit from `delegate`
         *  For example please look into `test_sid_delegate.cpp`
         *
         *  The typical usage looks like:
         *
         *  namespace foo_impl_ {
         *    template <class Sid>
         *    struct foo : delegate<Sid> { using delegate<Sid>::delegate; };
         *
         *    // some SID concept overloads for `foo` here
         *
         *    template <class Sid> foo<Sid> make_foo(Sid&& sid) { return {std::forward<Sid>(sid)}; }
         *  }
         *  using foo_impl_::make_foo;
         *
         *  Please be aware that in this example `delegate` could is instantiated with the reference if the L-value is
         *  passed to the `make_foo`. If R-value is passed, delegate is instantiated with the non-reference type.
         *  This affects the ownership semantic of the `foo` proxy. Example:
         *
         *  my_sid_t a = make_my_sid();
         *  auto foo_a = make_foo(a);  // `foo_a` contains the reference on `a`.
         *
         *  auto foo_b = make_foo(make_my_sid()); // `foo_b` contains the object of `my_sid`
         *
         *  Note also that the ctor of `delegate` is propagated to `foo` with C++11 `using` syntax.
         *  Please don't propagate `delagate` ctor using perfect forwarding like:
         *  template <class T> foo(T&& obj) : delegate<Sid>(std::forward<T>(obj)) {}
         *  It's a well known trap -- copy/move `foo` constructor could be shadowed that way.
         *  Although it is fine to do perfect forwarding with non unaray constructors. Like:
         *  template <class T> foo(T&& obj, int arg) : delegate<Sid>(std::forward<T>(obj)) { use_arg(arg); }
         *
         * @tparam Sid a object that models `SID` concept.
         */
        template <class Sid>
        struct delegate {
            static_assert(is_sid<Sid>::value, GT_INTERNAL_ERROR);
            Sid m_impl;

            template <bool IsRef = std::is_reference_v<Sid>, std::enable_if_t<!IsRef, int> = 0>
            delegate(Sid impl) : m_impl(std::move(impl)) {}

            template <bool IsRef = std::is_reference_v<Sid>, std::enable_if_t<IsRef, int> = 0>
            delegate(Sid impl) : m_impl(impl) {}
        };

        // Here and below SFINAE principal is used to ensure that the concept functions for the `delegate<Sid>` are
        // defined only if the corespondent functions are defined for `Sid`.

        template <class Sid>
        decltype(sid_get_origin(std::declval<Sid &>())) sid_get_origin(delegate<Sid> &obj) {
            return sid_get_origin(obj.m_impl);
        }

        template <class Sid>
        decltype(sid_get_ptr_diff(std::declval<Sid const &>())) sid_get_ptr_diff(delegate<Sid> const &);

        template <class Sid>
        decltype(sid_get_strides_kind(std::declval<Sid const &>())) sid_get_strides_kind(delegate<Sid> const &);

        template <class Sid>
        decltype(sid_get_strides(std::declval<Sid const &>())) sid_get_strides(delegate<Sid> const &obj) {
            return sid_get_strides(obj.m_impl);
        }
        template <class Sid>
        decltype(sid_get_lower_bounds(std::declval<Sid const &>())) sid_get_lower_bounds(delegate<Sid> const &obj) {
            return sid_get_lower_bounds(obj.m_impl);
        }
        template <class Sid>
        decltype(sid_get_upper_bounds(std::declval<Sid const &>())) sid_get_upper_bounds(delegate<Sid> const &obj) {
            return sid_get_upper_bounds(obj.m_impl);
        }

        template <class Arr, std::enable_if_t<std::is_array_v<Arr>, int> = 0>
        auto sid_get_origin(delegate<Arr &> &obj) {
            return sid::get_origin(obj.m_impl);
        }
        template <class Arr, std::enable_if_t<std::is_array_v<Arr>, int> = 0>
        auto sid_get_strides(delegate<Arr &> const &obj) {
            return get_strides(obj.m_impl);
        }
        template <class Arr, std::enable_if_t<std::is_array_v<Arr>, int> = 0>
        auto sid_get_lower_bounds(delegate<Arr &> const &obj) {
            return get_lower_bounds(obj.m_impl);
        }
        template <class Arr, std::enable_if_t<std::is_array_v<Arr>, int> = 0>
        auto sid_get_upper_bounds(delegate<Arr &> const &obj) {
            return get_upper_bounds(obj.m_impl);
        }
    } // namespace sid
} // namespace gridtools
