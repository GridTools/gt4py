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
#include "../meta/id.hpp"

namespace gridtools {
    namespace sid {

        enum class property { origin, strides, ptr_diff, strides_kind, lower_bounds, upper_bounds };

        template <property Property>
        using property_constant = std::integral_constant<property, Property>;

        namespace synthetic_impl_ {
            template <property Property, class T>
            struct mixin;

            template <class T>
            struct mixin<property::origin, T> {
                T m_val;
            };
            template <class T>
            T sid_get_origin(mixin<property::origin, T> const &obj) noexcept {
                return obj.m_val;
            }

            template <class T>
            struct mixin<property::strides, T> {
                T m_val;
            };
            template <class T>
            T sid_get_strides(mixin<property::strides, T> const &obj) noexcept {
                return obj.m_val;
            }

            template <class T>
            struct mixin<property::lower_bounds, T> {
                T m_val;
            };
            template <class T>
            T sid_get_lower_bounds(mixin<property::lower_bounds, T> const &obj) noexcept {
                return obj.m_val;
            }

            template <class T>
            struct mixin<property::upper_bounds, T> {
                T m_val;
            };
            template <class T>
            T sid_get_upper_bounds(mixin<property::upper_bounds, T> const &obj) noexcept {
                return obj.m_val;
            }

            template <class T>
            struct mixin<property::ptr_diff, T> {};
            template <class T>
            T sid_get_ptr_diff(mixin<property::ptr_diff, T> const &obj);

            template <class T>
            struct mixin<property::strides_kind, T> {};
            template <class T>
            T sid_get_strides_kind(mixin<property::strides_kind, T> const &);

            template <property>
            struct unique {};

            template <property Property, class T>
            struct unique_mixin : mixin<Property, T>, unique<Property> {
                unique_mixin() = default;
                unique_mixin(unique_mixin const &) = default;
                unique_mixin(unique_mixin &&) = default;
                unique_mixin &operator=(unique_mixin const &) = default;
                unique_mixin &operator=(unique_mixin &&) = default;

                template <class U>
                constexpr unique_mixin(U &&obj) noexcept : mixin<Property, T>{std::forward<U>(obj)} {}
            };

            template <class...>
            struct synthetic;

            template <>
            struct synthetic<> {
                template <property Property, class T>
                constexpr synthetic<unique_mixin<Property, T>> set() const &&noexcept {
                    return synthetic{};
                }

                template <property Property, class T>
                constexpr synthetic<unique_mixin<Property, std::decay_t<T>>> set(T &&val) const &&noexcept {
                    return {std::forward<T>(val), synthetic{}};
                }
            };

            template <class Mixin, class... Mixins>
            struct synthetic<Mixin, Mixins...> : Mixin, Mixins... {
                synthetic() = default;
                constexpr synthetic(synthetic<Mixins...> const &&src) noexcept : Mixins(std::move(src))... {}

                template <class T>
                constexpr synthetic(T &&val, synthetic<Mixins...> const &&src) noexcept
                    : Mixin{std::forward<T>(val)}, Mixins(std::move(src))... {}

                template <property Property, class T>
                constexpr synthetic<unique_mixin<Property, T>, Mixin, Mixins...> set(
                    meta::lazy::id<T> = {}, property_constant<Property> = {}) const &&noexcept {
                    return {std::move(*this)};
                }

                template <property Property, class T>
                constexpr synthetic<unique_mixin<Property, std::decay_t<T>>, Mixin, Mixins...> set(
                    T &&val, property_constant<Property> = {}) const &&noexcept {
                    return {std::forward<T>(val), std::move(*this)};
                }
            };
        } // namespace synthetic_impl_

        /**
         *  A tiny EDSL for creating SIDs from the parts described in the concept.
         *
         *  Usage:
         *
         *  \code
         *  auto my_sid = synthetic()
         *      .set<property::origin>(origin)
         *      .set<property::strides>(strides)
         *      .set<property::ptr_diff, ptr_diff>()
         *      .set<property::strides_kind, strides_kind>();
         *  \endcode
         *
         *  only `set<property::origin>` is required. Other `set`'s can be skipped.
         *  `set`'s can go in any order. `set` of the given property can participate at most once.
         *  Duplicated property `set`'s cause compiler error.
         *  Works both in run time and in compile time.
         *
         *  Due to CUDA8 bug `set` methods are supplied with default parameters of `property_constant` and
         *  `meta::lazy::id` types. This is needed when the type of `synthetic` is template dependant and we can't use
         *  `template` keyword (CUDA8 has problems in calculating `decltype` of expressions with `template`).
         *  In this case we have the possibility to write:
         *  ```
         *      expr
         *      .set(strides, property_constant<property::strides>)
         *      .set(meta::lazy::id<PtrDiff>(), property_constant<property::ptr_diff>());
         *  ```
         *  instead of
         *  ```
         *      expr
         *      .template set<property::strides>(strides)
         *      .template set<property::ptr_diff, PtrDiff>();
         *  ```
         *
         *  TODO(anstaf): remove this feature after CUDA8 drop.
         */
        constexpr synthetic_impl_::synthetic<> synthetic() { return {}; }
    } // namespace sid
} // namespace gridtools
