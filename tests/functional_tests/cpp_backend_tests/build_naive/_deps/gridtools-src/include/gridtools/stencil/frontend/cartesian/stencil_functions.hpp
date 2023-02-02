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

#include "../../../common/hymap.hpp"
#include "../../../common/tuple.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../common/dim.hpp"
#include "../../core/interval.hpp"
#include "accessor.hpp"
#include "expressions/expr_base.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace call_interfaces_impl_ {
                template <class Functor, class Region, class Eval>
                GT_FUNCTION std::enable_if_t<!std::is_void_v<Region>> call_functor(Eval eval) {
                    Functor::template apply<Eval &>(eval, Region());
                }

                // overload for the default interval (Functor with one argument)
                template <class Functor, class Region, class Eval>
                GT_FUNCTION std::enable_if_t<std::is_void_v<Region>> call_functor(Eval eval) {
                    Functor::template apply<Eval &>(eval);
                }

                template <class Key>
                struct sum_offset_generator_f {

                    using default_t = integral_constant<int_t, 0>;

                    template <class Lhs, class Rhs>
                    GT_FUNCTION auto operator()(Lhs &&lhs, Rhs &&rhs) const {
                        return host_device::at_key_with_default<Key, default_t>(std::forward<Lhs>(lhs)) +
                               host_device::at_key_with_default<Key, default_t>(std::forward<Rhs>(rhs));
                    }
                };

                template <class Res, class Lhs, class Rhs>
                GT_FUNCTION Res sum_offsets(Lhs &&lhs, Rhs rhs) {
                    using generators_t = meta::transform<sum_offset_generator_f, get_keys<Res>>;
                    return tuple_util::host_device::generate<generators_t, Res>(std::forward<Lhs>(lhs), std::move(rhs));
                }

                template <int_t I, int_t J, int_t K, class Accessor>
                GT_FUNCTION std::enable_if_t<I == 0 && J == 0 && K == 0, Accessor> get_offsets(Accessor acc) {
                    return acc;
                }

                template <int_t I,
                    int_t J,
                    int_t K,
                    class Accessor,
                    class Res = array<int_t, tuple_util::size<Accessor>::value>>
                GT_FUNCTION std::enable_if_t<I != 0 || J != 0 || K != 0, Res> get_offsets(Accessor acc) {
                    using offset_t = hymap::keys<dim::i, dim::j, dim::k>::
                        values<integral_constant<int_t, I>, integral_constant<int_t, J>, integral_constant<int_t, K>>;
                    return sum_offsets<Res>(std::move(acc), offset_t());
                }

                template <int_t I, int_t J, int_t K, class Params, class Eval, class Args>
                struct evaluator {
                    Eval &m_eval;
                    Args m_args;

                    template <class Accessor,
                        class Arg = std::decay_t<meta::at<Args, typename Accessor::index_t>>,
                        class Param = meta::at<Params, typename Accessor::index_t>,
                        std::enable_if_t<is_accessor<Accessor>::value && is_accessor<Arg>::value &&
                                             !(Param::intent_v == intent::inout && Arg::intent_v == intent::in),
                            int> = 0>
                    GT_FUNCTION decltype(auto) operator()(Accessor acc) const {
                        return m_eval(sum_offsets<Arg>(
                            get_offsets<I, J, K>(tuple_util::host_device::get<Accessor::index_t::value>(m_args)),
                            std::move(acc)));
                    }

                    template <class Accessor,
                        class Arg = std::decay_t<meta::at<Args, typename Accessor::index_t>>,
                        class Param = meta::at<Params, typename Accessor::index_t>,
                        std::enable_if_t<is_accessor<Accessor>::value && !is_accessor<Arg>::value &&
                                             !(Param::intent_v == intent::inout &&
                                                 std::is_const<std::remove_reference_t<Arg>>::value),
                            int> = 0>
                    GT_FUNCTION decltype(auto) operator()(Accessor) const {
                        return tuple_util::host_device::get<Accessor::index_t::value>(m_args);
                    }

                    template <class Op, class... Ts>
                    GT_FUNCTION auto operator()(expr<Op, Ts...> arg) const {
                        return expressions::evaluation::value(*this, std::move(arg));
                    }
                };

                template <int_t I, int_t J, int_t K, class Params, class Eval, class Args>
                constexpr GT_FUNCTION evaluator<I, J, K, Params, Eval, Args> make_evaluator(Eval &eval, Args args) {
                    return {eval, std::move(args)};
                }

                template <class Functor, class Region, int_t I, int_t J, int_t K, class Eval, class Args>
                GT_FUNCTION void evaluate_bound_functor(Eval &eval, Args args) {
                    call_functor<Functor, Region>(
                        make_evaluator<I, J, K, typename Functor::param_list>(eval, std::move(args)));
                }

                template <class Eval, class Arg, bool = is_accessor<Arg>::value>
                struct deduce_result_type : std::decay<decltype(std::declval<Eval &>()(std::declval<Arg &&>()))> {};

                template <class Eval, class Arg>
                struct deduce_result_type<Eval, Arg, false> : meta::lazy::id<Arg> {};

                /**
                 * @brief Use forced return type (if not void) or deduce the return type.
                 */
                template <class Eval, class ReturnType, class Arg, class...>
                struct get_result_type : std::conditional<std::is_void_v<ReturnType>,
                                             typename deduce_result_type<Eval, Arg>::type,
                                             ReturnType> {};

                template <class Accessor>
                using is_out_param = std::bool_constant<Accessor::intent_v == intent::inout>;
            } // namespace call_interfaces_impl_

            /** Main interface for calling stencil operators as functions.

                Usage: call<functor, region>::[return_type<>::][at<offseti, offsetj, offsetk>::]with(eval,
               accessors...);

                \tparam Functos The stencil operator to be called
                \tparam Region The region in which to call it (to take the proper overload). A region with no exact
               match is not called and will result in compilation error. The user is responsible for calling the proper
               apply overload) \tparam ReturnType Can be set or will be deduced from the first input argument \tparam
               Offi Offset along the i-direction (usually modified using at<...>) \tparam Offj Offset along the
               j-direction \tparam Offk Offset along the k-direction
            */
            template <class Functor,
                class Region = void,
                class ReturnType = void,
                int_t OffI = 0,
                int_t OffJ = 0,
                int_t OffK = 0>
            class call {
                static_assert(meta::is_instantiation_of<core::interval, Region>::value or std::is_void_v<Region>,
                    "Region should be a valid interval tag or void (default interval) to select the apply "
                    "specialization "
                    "in "
                    "the called stencil function");

                using params_t = typename Functor::param_list;
                using out_params_t = meta::filter<call_interfaces_impl_::is_out_param, params_t>;

                static_assert(meta::length<out_params_t>::value == 1,
                    "Trying to invoke stencil operator with more than one output as a function");

                using out_param_t = meta::first<out_params_t>;
                static constexpr size_t out_param_index = out_param_t::index_t::value;

              public:
                /** This alias is used to move the computation at a certain offset
                 */
                template <int_t I, int_t J, int_t K>
                using at = call<Functor, Region, ReturnType, I, J, K>;

                /**
                 * @brief alias to set the return type, e.g.
                 */
                template <typename ForcedReturnType>
                using return_type = call<Functor, Region, ForcedReturnType, OffI, OffJ, OffK>;

                /**
                 * With this interface a stencil function can be invoked and the offsets specified in the passed
                 * accessors are used to access values, w.r.t the offsets specified in a optional  at<..> statement.
                 */
                template <class Eval,
                    class... Args,
                    class Res =
                        typename call_interfaces_impl_::get_result_type<Eval, ReturnType, std::decay_t<Args>...>::type,
                    std::enable_if_t<sizeof...(Args) + 1 == meta::length<params_t>::value, int> = 0>
                GT_FUNCTION static Res with(Eval &eval, Args &&...args) {
                    Res res;
                    call_interfaces_impl_::evaluate_bound_functor<Functor, Region, OffI, OffJ, OffK>(eval,
                        tuple_util::host_device::insert<out_param_index>(
                            res, tuple<Args &&...>{std::forward<Args>(args)...}));
                    return res;
                }
            };

            /**
             * Main interface for calling stencil operators as functions with side-effects. The interface accepts a list
             * of arguments to be passed to the called function and these arguments can be accessors or simple values.
             *
             * Usage : call_proc<functor, region>::[at<offseti, offsetj, offsetk>::]with(eval, accessors_or_values...);
             *
             * Accessors_or_values referes to a list of arguments that may be accessors of the caller functions or local
             * variables of the type accessed (or converted to) by the accessor in the called function, where the
             * results should be obtained from. The values can also be used by the function as inputs.
             *
             * \tparam Functor The stencil operator to be called
             * \tparam Region The region in which to call it (to take the proper overload). A region with no exact match
             * is not called and will result in compilation error. The user is responsible for calling the proper apply
             * overload) \tparam OffI Offset along the i-direction (usually modified using at<...>) \tparam OffJ Offset
             * along the j-direction \tparam OffK Offset along the k-direction
             * */
            template <class Functor, class Region = void, int_t OffI = 0, int_t OffJ = 0, int_t OffK = 0>
            struct call_proc {

                static_assert(meta::is_instantiation_of<core::interval, Region>::value or std::is_void_v<Region>,
                    "Region should be a valid interval tag or void (default interval) to select the apply "
                    "specialization "
                    "in "
                    "the called stencil function");

                /** This alias is used to move the computation at a certain offset
                 */
                template <int_t I, int_t J, int_t K>
                using at = call_proc<Functor, Region, I, J, K>;

                /**
                 * With this interface a stencil function can be invoked and the offsets specified in the passed
                 * accessors are used to access values, w.r.t the offsets specified in a optional at<..> statement.
                 */
                template <class Eval, class... Args>
                GT_FUNCTION static std::enable_if_t<sizeof...(Args) ==
                                                    meta::length<typename Functor::param_list>::value>
                with(Eval &eval, Args &&...args) {
                    call_interfaces_impl_::evaluate_bound_functor<Functor, Region, OffI, OffJ, OffK>(
                        eval, tuple<Args &&...>{std::forward<Args>(args)...});
                }
            };
        } // namespace cartesian
    }     // namespace stencil
} // namespace gridtools
