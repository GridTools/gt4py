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

#include <cassert>
#include <type_traits>
#include <utility>

#include "../../common/for_each.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../../sid/concept.hpp"
#include "../common/caches.hpp"
#include "../common/dim.hpp"
#include "../common/extent.hpp"
#include "../common/intent.hpp"
#include "../core/backend.hpp"
#include "../core/cache_info.hpp"
#include "../core/compute_extents_metafunctions.hpp"
#include "../core/esf.hpp"
#include "../core/execution_types.hpp"
#include "../core/functor_metafunctions.hpp"
#include "../core/is_tmp_arg.hpp"
#include "../core/mss.hpp"

namespace gridtools {
    namespace stencil {
        namespace frontend_impl_ {
            template <class, class = void>
            struct has_param_list : std::false_type {};

            template <class T>
            struct has_param_list<T, std::void_t<typename T::param_list>> : std::true_type {};

            template <class, class = void>
            struct is_accessor : std::false_type {};

            template <class T>
            struct is_accessor<T,
                std::enable_if_t<is_extent<typename T::extent_t>::value &&
                                 std::is_convertible_v<decltype(T::index_t::value), size_t> &&
                                 std::is_same_v<intent, std::decay_t<decltype(T::intent_v)>>>> : std::true_type {};

            template <class Param>
            using param_index = std::integral_constant<size_t, Param::index_t::value>;

            template <class ParamList,
                class Actual = meta::rename<meta::list, meta::transform<param_index, ParamList>>,
                class Expected = meta::make_indices_for<ParamList>>
            using check_param_list = std::is_same<Actual, Expected>;

            template <class F, class... Args>
            struct validate_functor {
                static_assert(has_param_list<F>::value,
                    "The type param_list was not found in a user functor definition. All user functors must have a "
                    "type alias called `param_list`, which is an instantiation of `make_param_list` accessors "
                    "defined in the functor Example:\n using v1=in_accessor<0>;\n using v2=inout_accessor<1>;\n "
                    "using param_list=make_param_list<v1, v2>;\n");

                static_assert(meta::is_list<typename F::param_list>::value,
                    "`param_list` must be a type list (use `make_param_list`).");

                static_assert(meta::all_of<is_accessor, typename F::param_list>::value,
                    "All members of `param_list` must be `accessor`s.");

                static_assert(check_param_list<typename F::param_list>::value,
                    "The `accessor`s in `param_list` of a stencil operator"
                    "don't have increasing index");

                static_assert(meta::length<typename F::param_list>::value == sizeof...(Args),
                    "The number of actual arguments should match the number of parameters.");

                using type = F;
            };

            template <class...>
            struct spec {};

            template <class ExecutionType, class Esfs, class Caches>
            struct spec<core::mss_descriptor<ExecutionType, Esfs, Caches>> {
                template <class F, class... Args>
                constexpr spec<core::mss_descriptor<ExecutionType,
                    meta::push_back<Esfs,
                        core::esf_descriptor<typename validate_functor<F, Args...>::type, meta::list<Args...>, void>>,
                    Caches>>
                stage(F, Args...) const {
                    return {};
                }

                template <int_t IMinus, int_t IPlus, int_t JMinus, int_t JPlus, class F, class... Args>
                constexpr spec<core::mss_descriptor<ExecutionType,
                    meta::push_back<Esfs,
                        core::esf_descriptor<typename validate_functor<F, Args...>::type,
                            meta::list<Args...>,
                            extent<IMinus, IPlus, JMinus, JPlus>>>,
                    Caches>>
                stage_with_extent(extent<IMinus, IPlus, JMinus, JPlus>, F, Args...) const {
                    return {};
                }
            };

            template <class ExecutionType, class... Caches>
            struct empty_spec : spec<core::mss_descriptor<ExecutionType, meta::list<>, meta::list<Caches...>>> {
                template <class... Args>
                constexpr empty_spec<ExecutionType, Caches..., core::cache_info<Args, meta::list<cache_type::ij>>...>
                ij_cached(Args...) const {
                    static_assert(meta::is_set<meta::list<Args...>>::value, "Duplicated arguments.");
                    static_assert(
                        meta::is_set<meta::list<typename Caches::plh_t..., Args...>>::value, "Duplicated caches.");
                    static_assert(
                        std::conjunction<core::is_tmp_arg<Args>...>::value, "Only temporary args can be IJ-cached.");
                    return {};
                }
                template <class... Args>
                constexpr empty_spec<ExecutionType, Caches..., core::cache_info<Args, meta::list<cache_type::k>>...>
                k_cached(Args...) const {
                    static_assert(meta::is_set<meta::list<Args...>>::value, "Duplicated arguments.");
                    static_assert(
                        meta::is_set<meta::list<typename Caches::plh_t..., Args...>>::value, "Duplicated caches.");
                    static_assert(std::conjunction<core::is_tmp_arg<Args>...>::value,
                        "Only temporary args can be K-cached without fill or flush policies.");
                    return {};
                }
                template <class... Args>
                constexpr empty_spec<ExecutionType,
                    Caches...,
                    core::cache_info<Args, meta::list<cache_type::k>, meta::list<cache_io_policy::flush>>...>
                k_cached(cache_io_policy::flush, Args...) const {
                    static_assert(meta::is_set<meta::list<Args...>>::value, "Duplicated arguments.");
                    static_assert(
                        meta::is_set<meta::list<typename Caches::plh_t..., Args...>>::value, "Duplicated caches.");
                    return {};
                }
                template <class... Args>
                constexpr empty_spec<ExecutionType,
                    Caches...,
                    core::cache_info<Args, meta::list<cache_type::k>, meta::list<cache_io_policy::fill>>...>
                k_cached(cache_io_policy::fill, Args...) const {
                    static_assert(meta::is_set<meta::list<Args...>>::value, "Duplicated arguments.");
                    static_assert(
                        meta::is_set<meta::list<typename Caches::plh_t..., Args...>>::value, "Duplicated caches.");
                    return {};
                }
                template <class... Args>
                constexpr empty_spec<ExecutionType,
                    Caches...,
                    core::cache_info<Args,
                        meta::list<cache_type::k>,
                        meta::list<cache_io_policy::fill, cache_io_policy::flush>>...>
                k_cached(cache_io_policy::fill, cache_io_policy::flush, Args...) const {
                    static_assert(meta::is_set<meta::list<Args...>>::value, "Duplicated arguments.");
                    static_assert(
                        meta::is_set<meta::list<typename Caches::plh_t..., Args...>>::value, "Duplicated caches.");
                    return {};
                }
                template <class... Args>
                constexpr empty_spec<ExecutionType,
                    Caches...,
                    core::cache_info<Args,
                        meta::list<cache_type::k>,
                        meta::list<cache_io_policy::fill, cache_io_policy::flush>>...>
                k_cached(cache_io_policy::flush, cache_io_policy::fill, Args...) const {
                    static_assert(meta::is_set<meta::list<Args...>>::value, "Duplicated arguments.");
                    static_assert(
                        meta::is_set<meta::list<typename Caches::plh_t..., Args...>>::value, "Duplicated caches.");
                    return {};
                }
            };

            constexpr empty_spec<core::parallel> execute_parallel() { return {}; }
            constexpr empty_spec<core::forward> execute_forward() { return {}; }
            constexpr empty_spec<core::backward> execute_backward() { return {}; }

            template <class Mss, class... Msses>
            constexpr spec<Mss, Msses...> multi_pass(spec<Mss>, spec<Msses>...) {
                return {};
            }

            template <class... Ts>
            void multi_pass(Ts...) {
                static_assert(sizeof...(Ts) < 0, "Unexpected arguments of gridtools::stencil::multi_pass.");
            }

            template <size_t I>
            struct arg : std::integral_constant<size_t, I> {};

            template <class Interval>
            struct check_valid_apply_overloads {
                template <class Functor>
                using apply = core::check_valid_apply_overloads<Functor, Interval>;
            };

            template <class Comp, class Backend, class Grid, class... Fields, size_t... Is>
            auto run_impl(Comp comp, Backend &&be, Grid const &grid, std::index_sequence<Is...>, Fields &&...fields)
                -> std::void_t<decltype(comp(arg<Is>()...))> {
                using spec_t = decltype(comp(arg<Is>()...));
                static_assert(
                    meta::is_instantiation_of<spec, spec_t>::value, "Invalid stencil composition specification.");
                static_assert(
                    meta::is_instantiation_of<core::interval, typename Grid::interval_t>::value, "Invalid grid.");
                using functors_t = meta::transform<meta::first, meta::flatten<meta::transform<meta::second, spec_t>>>;
                static_assert(meta::all_of<check_valid_apply_overloads<typename Grid::interval_t>::template apply,
                                  functors_t>::value,
                    "Invalid stencil operator detected.");

                using data_store_map_t = typename hymap::keys<arg<Is>...>::template values<Fields &...>;
#ifndef NDEBUG
                using extent_map_t = core::get_extent_map_from_msses<spec_t>;
                auto check_bounds = [origin = grid.origin(), size = grid.size()](auto arg, auto const &field) {
                    using extent_t = core::lookup_extent_map<extent_map_t, decltype(arg)>;
                    // There is no check in k-direction because at the fields may be used within subintervals
                    // TODO(anstaf): find the proper place to check k-bounds
                    for_each<meta::list<dim::i, dim::j>>(
                        [&, l_bounds = sid::get_lower_bounds(field), u_bounds = sid::get_upper_bounds(field)](auto d) {
                            using dim_t = decltype(d);
                            assert(at_key<dim_t>(origin) + extent_t::minus(d) >= sid::get_lower_bound<dim_t>(l_bounds));
                            assert(at_key<dim_t>(origin) + at_key<dim_t>(size) + extent_t::plus(d) <=
                                   sid::get_upper_bound<dim_t>(u_bounds));
                        });
                    return 0;
                };
                using loop_t = int[sizeof...(Is)];
                (void)loop_t{check_bounds(arg<Is>(), fields)...};
#endif
                core::call_entry_point_f<spec_t>()(std::forward<Backend>(be), grid, data_store_map_t{fields...});
            }

            template <class... Ts>
            void run_impl(Ts...) {
                static_assert(sizeof...(Ts) < 0, "Unexpected first argument of gridtools::stencil::run.");
            }

            template <class Comp, class Backend, class Grid, class... Fields>
            void run(Comp comp, Backend &&be, Grid const &grid, Fields &&...fields) {
                static_assert(
                    std::conjunction<is_sid<Fields>...>::value, "All computation fields must satisfy SID concept.");
                run_impl(comp,
                    std::forward<Backend>(be),
                    grid,
                    std::index_sequence_for<Fields...>(),
                    std::forward<Fields>(fields)...);
            }

            template <class F, class Backend, class Grid, class... Fields>
            void run_single_stage(F, Backend &&be, Grid const &grid, Fields &&...fields) {
                return run([](auto... args) { return execute_parallel().stage(F(), args...); },
                    std::forward<Backend>(be),
                    grid,
                    std::forward<Fields>(fields)...);
            }

            template <class... Msses, class Arg>
            constexpr core::lookup_extent_map<core::get_extent_map_from_msses<spec<Msses...>>, Arg> get_arg_extent(
                spec<Msses...>, Arg) {
                return {};
            }

            template <class Mss>
            using rw_args_from_mss = core::compute_readwrite_args<typename Mss::esf_sequence_t>;

            template <class Msses,
                class RwArgsLists = meta::transform<rw_args_from_mss, Msses>,
                class RawRwArgs = meta::flatten<RwArgsLists>>
            using all_rw_args = meta::dedup<RawRwArgs>;

            template <class... Msses,
                class Arg,
                class RwPlhs = all_rw_args<spec<Msses...>>,
                intent Intent = meta::st_contains<RwPlhs, Arg>::value ? intent::inout : intent::in>
            constexpr std::integral_constant<intent, Intent> get_arg_intent(spec<Msses...>, Arg) {
                return {};
            }
        } // namespace frontend_impl_
        using frontend_impl_::execute_backward;
        using frontend_impl_::execute_forward;
        using frontend_impl_::execute_parallel;
        using frontend_impl_::get_arg_extent;
        using frontend_impl_::get_arg_intent;
        using frontend_impl_::multi_pass;
        using frontend_impl_::run;
        using frontend_impl_::run_single_stage;
    } // namespace stencil
} // namespace gridtools
