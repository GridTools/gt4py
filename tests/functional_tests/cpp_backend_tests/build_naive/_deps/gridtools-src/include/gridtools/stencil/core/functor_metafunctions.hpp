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

#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace functor_metafunctions_impl_ {
                struct probe {};

                template <class T, class = void>
                struct has_apply : std::false_type {};

                template <class T>
                struct has_apply<T, std::void_t<decltype(T::apply(std::declval<probe const &>()))>> : std::true_type {};

                template <class From>
                struct to_resolver {
                    template <class R, class To>
                    static To select(R (*)(probe const &, interval<From, To>));
                };

                template <class Functor, class From>
                using unsafe_resolve_to = decltype(to_resolver<From>::select(&Functor::template apply<probe const &>));

                template <class Functor, class From, class = void>
                struct find_interval_parameter {
                    using type = meta::list<>;
                };

                template <class Functor, class From>
                struct find_interval_parameter<Functor, From, std::void_t<unsafe_resolve_to<Functor, From>>> {
                    using type = meta::list<interval<From, unsafe_resolve_to<Functor, From>>>;
                };

                template <class Functor>
                struct find_interval_parameter_f {
                    template <class From>
                    using apply = typename find_interval_parameter<Functor, From>::type;
                };

                template <class Interval>
                struct interval_intersects_with {
                    using from_index_t = level_to_index<meta::first<Interval>>;
                    using to_index_t = level_to_index<meta::second<Interval>>;
                    static_assert(from_index_t::value <= to_index_t::value, GT_INTERNAL_ERROR);
                    template <class Other>
                    using apply = std::bool_constant<level_to_index<meta::first<Other>>::value <= to_index_t::value &&
                                                     level_to_index<meta::second<Other>>::value >= from_index_t::value>;
                };

                template <class From>
                struct make_level_from_index_f {
                    using from_index_t = level_to_index<From>;
                    template <class N>
                    using apply = index_to_level<level_index<N::value + from_index_t::value, From::offset_limit>>;
                };

                template <class Intervals, class Level>
                struct add_level {
                    using type = meta::list<meta::flatten<Intervals>, meta::list<interval<Level, Level>>>;
                };

                template <class Intervals, uint_t Splitter, int_t Limit, class Level>
                struct add_level<
                    meta::list<Intervals,
                        meta::list<interval<level<Splitter, Limit, Limit>, level<Splitter, Limit, Limit>>>>,
                    Level> {
                    using type = meta::list<meta::push_back<Intervals, interval<level<Splitter, Limit, Limit>, Level>>>;
                };

                template <class From, class To>
                using all_levels = meta::transform<make_level_from_index_f<From>::template apply,
                    meta::make_indices_c<level_to_index<To>::value - level_to_index<From>::value + 1>>;

                template <class Interval>
                using split_interval = meta::flatten<meta::foldl<meta::force<add_level>::apply,
                    meta::list<meta::list<>>,
                    all_levels<meta::first<Interval>, meta::second<Interval>>>>;

                template <class Functor, class Interval>
                using find_interval_parameters = meta::filter<interval_intersects_with<Interval>::template apply,
                    meta::flatten<meta::transform<find_interval_parameter_f<Functor>::template apply,
                        all_levels<index_to_level<level_index<0, Interval::offset_limit>>, meta::second<Interval>>>>>;

                template <class F, class Lhs, class Rhs>
                    struct intersection_detector
                    : std::bool_constant <
                      level_to_index<meta::second<Lhs>>::value<level_to_index<meta::first<Rhs>>::value> {
                    static_assert(intersection_detector<F, Lhs, Rhs>::value,
                        "A stencil operator with intersecting intervals was detected. Search above for "
                        "`intersection_detector` in this compiler error output to determine the stencil operator and "
                        "the intervals.");
                };

                template <class Functor, class IntervalParameters>
                struct has_any_apply
                    : std::bool_constant<has_apply<Functor>::value || meta::length<IntervalParameters>::value != 0> {
                    static_assert(has_any_apply<Functor, IntervalParameters>::value,
                        "A stencil operator without any apply() overload within the given interval.\nSearch above "
                        "for `has_any_apply` in this compiler error output to determine the functor and the interval.");
                };

                template <class F, class FullInterval, class Interval>
                struct is_from_level_valid
                    : std::bool_constant<meta::first<Interval>::offset != -Interval::offset_limit ||
                                         level_to_index<meta::first<Interval>>::value <=
                                             level_to_index<meta::first<FullInterval>>::value> {
                    static_assert(is_from_level_valid<F, FullInterval, Interval>::value,
                        "The interval `from` level offset could be equal to `-offset_limit` only if this level is less "
                        "or equal to the `from` level of the full computation interval.\nSearch above for "
                        "`is_from_level_valid` in this compiler error output to determine the functor and the "
                        "interval.");
                };

                template <class F, class FullInterval, class Interval>
                struct is_to_level_valid
                    : std::bool_constant<meta::second<Interval>::offset != Interval::offset_limit ||
                                         level_to_index<meta::second<Interval>>::value >=
                                             level_to_index<meta::second<FullInterval>>::value> {
                    static_assert(is_to_level_valid<F, FullInterval, Interval>::value,
                        "The interval `to` level offset could be equal to `offset_limit` only if this level is greater "
                        "or equal to the `to` level of the full computation interval.\nSearch above for "
                        "`is_to_level_valid` in this compiler error output to determine the functor and the interval.");
                };

                template <template <class...> class F, class L>
                struct transform_neighbours;

                template <template <class...> class F, template <class...> class L>
                struct transform_neighbours<F, L<>> {
                    using type = L<>;
                };

                template <template <class...> class F, template <class...> class L, class T>
                struct transform_neighbours<F, L<T>> {
                    using type = L<>;
                };

                template <template <class...> class F, template <class...> class L, class T0, class T1, class... Ts>
                struct transform_neighbours<F, L<T0, T1, Ts...>> {
                    using type = meta::push_front<typename transform_neighbours<F, L<T1, Ts...>>::type, F<T0, T1>>;
                };

                // if overloads are valid this alias evaluate to std::true_type
                // otherwise static_assert is triggered.
                template <class Functor,
                    class Interval,
                    class IntervalParameters = find_interval_parameters<Functor, Interval>,
                    class HasAnyApply = has_any_apply<Functor, IntervalParameters>,
                    class IntersectionDetectors =
                        typename transform_neighbours<meta::curry<intersection_detector, Functor>::template apply,
                            IntervalParameters>::type,
                    class FromLevelValidators =
                        meta::transform<meta::curry<is_from_level_valid, Functor, Interval>::template apply,
                            IntervalParameters>,
                    class ToLevelValidators =
                        meta::transform<meta::curry<is_to_level_valid, Functor, Interval>::template apply,
                            IntervalParameters>>
                using check_valid_apply_overloads = meta::all<
                    meta::push_back<meta::concat<IntersectionDetectors, FromLevelValidators, ToLevelValidators>,
                        HasAnyApply>>;

                template <class Index, class Intervals>
                struct find_in_interval_parameters;

                template <class Index, template <class...> class L>
                struct find_in_interval_parameters<Index, L<>> {
                    using type = meta::list<>;
                };

                template <class Index, template <class...> class L, class Interval, class... Intervals>
                struct find_in_interval_parameters<Index, L<Interval, Intervals...>> {
                    using type = typename meta::if_c<(level_to_index<meta::first<Interval>>::value > Index::value),
                        meta::list<>,
                        meta::if_c<(level_to_index<meta::second<Interval>>::value >= Index::value),
                            meta::list<Interval>,
                            find_in_interval_parameters<Index, L<Intervals...>>>>::type;
                };

                template <class Key,
                    class Functor,
                    class Interval,
                    class Params = typename find_in_interval_parameters<level_to_index<meta::first<Key>>,
                        find_interval_parameters<Functor, Interval>>::type,
                    bool HasApply = has_apply<Functor>::value>
                struct make_functor_map_item;

                template <class Key, class Functor, class Interval>
                struct make_functor_map_item<Key, Functor, Interval, meta::list<>, false> {
                    using type = meta::list<Key>;
                };

                template <class Key, class Functor, class Interval>
                struct make_functor_map_item<Key, Functor, Interval, meta::list<>, true> {
                    using type = meta::list<Key, Functor>;
                };

                template <class Functor, class Param>
                struct bound_functor : Functor {
                    template <class Eval>
                    static GT_FUNCTION void apply(Eval &&eval) {
                        Functor::apply(std::forward<Eval>(eval), Param());
                    }
                };

                template <class Key, class Functor, class Interval, class Param, bool HasApply>
                struct make_functor_map_item<Key, Functor, Interval, meta::list<Param>, HasApply> {
                    using type = meta::list<Key, bound_functor<Functor, Param>>;
                };

                template <class Functor, class Interval>
                struct item_maker_f {
                    template <class Key>
                    using apply = typename make_functor_map_item<Key, Functor, Interval>::type;
                };

                template <class Functor, class Interval>
                using make_functor_map =
                    meta::transform<item_maker_f<Functor, Interval>::template apply, split_interval<Interval>>;

            } // namespace functor_metafunctions_impl_
            using functor_metafunctions_impl_::bound_functor;
            using functor_metafunctions_impl_::check_valid_apply_overloads;
            using functor_metafunctions_impl_::make_functor_map;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
