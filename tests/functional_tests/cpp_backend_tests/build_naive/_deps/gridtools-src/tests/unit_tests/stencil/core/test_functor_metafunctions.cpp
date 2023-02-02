/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/core/functor_metafunctions.hpp>
#include <gridtools/stencil/core/interval.hpp>
#include <gridtools/stencil/core/level.hpp>

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace {
                // Note that the contract for `check_valid_apply_overloads` is that it fails to compile
                // if overloads are invalid.
                // This is for practical reason -- compliation error idicates what is wrong with particaular overload.
                // Hence automatic unit test is only able to ensure that there are no false negatives.

                // The cases with commneted out asserts are for false positives testing. It can be done only manualy:
                // uncommnent assert and ensure that the error message indicates the right problem.

                using interval_t = interval<level<0, 1, 3>, level<2, -1, 3>>;

                struct empty {};
                //                static_assert(check_valid_apply_overloads<empty, interval_t>::value);

                struct simple {
                    template <class T>
                    static void apply(T);
                };
                static_assert(check_valid_apply_overloads<simple, interval_t>::value);

                using expected_simple_map_t = meta::list<meta::list<interval<level<0, 1, 3>, level<0, 1, 3>>, simple>,
                    meta::list<interval<level<0, 2, 3>, level<0, 2, 3>>, simple>,
                    meta::list<interval<level<0, 3, 3>, level<1, -3, 3>>, simple>,
                    meta::list<interval<level<1, -2, 3>, level<1, -2, 3>>, simple>,
                    meta::list<interval<level<1, -1, 3>, level<1, -1, 3>>, simple>,
                    meta::list<interval<level<1, 1, 3>, level<1, 1, 3>>, simple>,
                    meta::list<interval<level<1, 2, 3>, level<1, 2, 3>>, simple>,
                    meta::list<interval<level<1, 3, 3>, level<2, -3, 3>>, simple>,
                    meta::list<interval<level<2, -2, 3>, level<2, -2, 3>>, simple>,
                    meta::list<interval<level<2, -1, 3>, level<2, -1, 3>>, simple>>;

                static_assert(std::is_same_v<make_functor_map<simple, interval_t>, expected_simple_map_t>);

                struct good {
                    template <class T>
                    static void apply(T);
                    template <class T>
                    static void apply(T, interval<level<0, -2, 3>, level<0, 1, 3>>);
                    template <class T>
                    static void apply(T, interval<level<0, 3, 3>, level<1, 1, 3>>);
                    template <class T>
                    static void apply(T, interval<level<1, 2, 3>, level<2, -3, 3>>);
                };
                static_assert(check_valid_apply_overloads<good, interval_t>::value);

                using expected_good_map_t = meta::list<
                    meta::list<interval<level<0, 1, 3>, level<0, 1, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, -2, 3>, level<0, 1, 3>>>>,
                    meta::list<interval<level<0, 2, 3>, level<0, 2, 3>>, good>,
                    meta::list<interval<level<0, 3, 3>, level<1, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, -2, 3>, level<1, -2, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, -1, 3>, level<1, -1, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, 1, 3>, level<1, 1, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, 2, 3>, level<1, 2, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<1, 2, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<1, 3, 3>, level<2, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<1, 2, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<2, -2, 3>, level<2, -2, 3>>, good>,
                    meta::list<interval<level<2, -1, 3>, level<2, -1, 3>>, good>>;

                static_assert(std::is_same_v<make_functor_map<good, interval_t>, expected_good_map_t>);

                struct intersect {
                    template <class T>
                    static void apply(T, interval<level<0, -2, 3>, level<1, 2, 3>>);
                    template <class T>
                    static void apply(T, interval<level<1, -2, 3>, level<2, -3, 3>>);
                };
                //                static_assert(check_valid_apply_overloads<intersect, interval_t>::value);

                struct gaps {
                    template <class T>
                    static void apply(T, interval<level<0, 3, 3>, level<1, -3, 3>>);
                    template <class T>
                    static void apply(T, interval<level<1, 3, 3>, level<2, -3, 3>>);
                };
                static_assert(check_valid_apply_overloads<gaps, interval_t>::value);

                using expected_gaps_map_t = meta::list<meta::list<interval<level<0, 1, 3>, level<0, 1, 3>>>,
                    meta::list<interval<level<0, 2, 3>, level<0, 2, 3>>>,
                    meta::list<interval<level<0, 3, 3>, level<1, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<gaps, interval<level<0, 3, 3>, level<1, -3, 3>>>>,
                    meta::list<interval<level<1, -2, 3>, level<1, -2, 3>>>,
                    meta::list<interval<level<1, -1, 3>, level<1, -1, 3>>>,
                    meta::list<interval<level<1, 1, 3>, level<1, 1, 3>>>,
                    meta::list<interval<level<1, 2, 3>, level<1, 2, 3>>>,
                    meta::list<interval<level<1, 3, 3>, level<2, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<gaps, interval<level<1, 3, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<2, -2, 3>, level<2, -2, 3>>>,
                    meta::list<interval<level<2, -1, 3>, level<2, -1, 3>>>>;
                static_assert(std::is_same_v<make_functor_map<gaps, interval_t>, expected_gaps_map_t>);

                struct good_intervals {
                    template <class T>
                    static void apply(T, interval<level<1, 2, 3>, level<1, 2, 3>>);
                    template <class T>
                    static void apply(T, interval<level<0, -3, 3>, level<0, -3, 3>>);
                    template <class T>
                    static void apply(T, interval<level<2, 3, 3>, level<2, 3, 3>>);
                };
                static_assert(
                    check_valid_apply_overloads<good_intervals, interval<level<0, -3, 3>, level<2, 2, 3>>>::value);

                struct bad_from_level {
                    template <class T>
                    static void apply(T, interval<level<1, -3, 3>, level<4, 1, 3>>);
                };
                //                static_assert(check_valid_apply_overloads<bad_from_level, interval_t>::value);

                struct bad_to_level {
                    template <class T>
                    static void apply(T, interval<level<1, 1, 3>, level<1, 3, 3>>);
                };
                //                static_assert(check_valid_apply_overloads<bad_to_level, interval_t>::value);
            } // namespace
        }     // namespace core
    }         // namespace stencil
} // namespace gridtools
