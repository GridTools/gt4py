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

#include "curry_fun.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Classic folds.
         */
        namespace lazy {
            template <template <class...> class, class...>
            struct foldl;
            template <template <class...> class, class...>
            struct foldr;
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(foldl, (template <class...> class F, class... Args), (F, Args...));
        GT_META_DELEGATE_TO_LAZY(foldr, (template <class...> class F, class... Args), (F, Args...));

        namespace lazy {
            template <template <class...> class F>
            struct foldl<F> {
                using type = curry_fun<meta::foldl, F>;
            };
            template <template <class...> class F, class S, template <class...> class L>
            struct foldl<F, S, L<>> {
                using type = S;
            };
            template <template <class...> class F, class S, template <class...> class L, class T>
            struct foldl<F, S, L<T>> {
                using type = F<S, T>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2>
            struct foldl<F, S, L<T1, T2>> {
                using type = F<F<S, T1>, T2>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2, class T3>
            struct foldl<F, S, L<T1, T2, T3>> {
                using type = F<F<F<S, T1>, T2>, T3>;
            };
            template <template <class...> class F,
                class S,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class T4>
            struct foldl<F, S, L<T1, T2, T3, T4>> {
                using type = F<F<F<F<S, T1>, T2>, T3>, T4>;
            };
            template <template <class...> class F>
            struct foldr<F> {
                using type = curry_fun<meta::foldr, F>;
            };
            template <template <class...> class F, class S, template <class...> class L>
            struct foldr<F, S, L<>> {
                using type = S;
            };
            template <template <class...> class F, class S, template <class...> class L, class T>
            struct foldr<F, S, L<T>> {
                using type = F<S, T>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2>
            struct foldr<F, S, L<T1, T2>> {
                using type = F<F<S, T2>, T1>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2, class T3>
            struct foldr<F, S, L<T1, T2, T3>> {
                using type = F<F<F<S, T3>, T2>, T1>;
            };
            template <template <class...> class F,
                class S,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class T4>
            struct foldr<F, S, L<T1, T2, T3, T4>> {
                using type = F<F<F<F<S, T4>, T3>, T2>, T1>;
            };
            namespace fold_impl_ {
                template <class>
                struct id;
                template <class>
                struct state;
                template <template <class...> class, class>
                struct folder;
                template <template <class...> class F, class S>
                struct state<folder<F, S> &&> {
                    using type = S;
                };
                template <template <class...> class F, class S, class T>
                folder<F, F<S, T>> &&operator+(folder<F, S> &&, id<T> *);
                template <template <class...> class F, class S, class T>
                folder<F, F<S, T>> &&operator+(id<T> *, folder<F, S> &&);
            } // namespace fold_impl_

            template <template <class...> class F,
                class S,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class T5,
                class... Ts>
            struct foldl<F, S, L<T1, T2, T3, T4, T5, Ts...>>
                : fold_impl_::state<decltype(
                      (std::declval<fold_impl_::folder<F, F<F<F<F<F<S, T1>, T2>, T3>, T4>, T5>> &&>() + ... +
                          (fold_impl_::id<Ts> *)0))> {};

            template <template <class...> class F,
                class S,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class T5,
                class... Ts>
            struct foldr<F, S, L<T1, T2, T3, T4, T5, Ts...>> {
                using type =
                    F<F<F<F<F<typename fold_impl_::state<decltype(
                                  ((fold_impl_::id<Ts> *)0 + ... + std::declval<fold_impl_::folder<F, S> &&>()))>::type,
                                T5>,
                              T4>,
                            T3>,
                          T2>,
                        T1>;
            };
        } // namespace lazy
    }     // namespace meta
} // namespace gridtools
