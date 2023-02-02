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

#include "../../../../common/host_device.hpp"
#include "../../../../meta.hpp"
#include "../accessor.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            /** Expressions Definition
                This is the base class of a binary expression, containing the instances of the two arguments.
                The expression should be a static constexpr object, instantiated once for all at the beginning of the
               run.
            */
            template <class Op, class... Args>
            struct expr;

            template <class Op, class Arg>
            struct expr<Op, Arg> {
                Arg m_arg;
            };

            template <class Op, class Lhs, class Rhs>
            struct expr<Op, Lhs, Rhs> {
                Lhs m_lhs;
                Rhs m_rhs;
            };

            namespace expressions {
                template <class>
                struct is_expr : std::false_type {};
                template <class... Ts>
                struct is_expr<expr<Ts...>> : std::true_type {};

                template <class Arg>
                using expr_or_accessor = std::bool_constant<is_expr<Arg>::value || is_accessor<Arg>::value>;

                template <class Op,
                    class... Args,
                    std::enable_if_t<std::disjunction<expr_or_accessor<Args>...>::value, int> = 0>
                GT_FUNCTION constexpr expr<Op, Args...> make_expr(Op, Args... args) {
                    return {args...};
                }

                namespace evaluation {
                    template <class Eval, class Arg, std::enable_if_t<std::is_arithmetic_v<Arg>, int> = 0>
                    GT_FUNCTION constexpr Arg apply_eval(Eval &&, Arg arg) {
                        return arg;
                    }

                    template <class Eval, class Arg, std::enable_if_t<!std::is_arithmetic_v<Arg>, int> = 0>
                    GT_FUNCTION constexpr decltype(auto) apply_eval(Eval &&eval, Arg arg) {
                        return std::forward<Eval>(eval)(std::move(arg));
                    }

                    template <class Eval, class Op, class Arg>
                    GT_FUNCTION constexpr auto value(Eval &&eval, expr<Op, Arg> arg) {
                        return Op()(std::forward<Eval>(eval)(std::move(arg.m_arg)));
                    }

                    template <class Eval, class Op, class Lhs, class Rhs>
                    GT_FUNCTION constexpr auto value(Eval &&eval, expr<Op, Lhs, Rhs> arg) {
                        return Op()(apply_eval(eval, std::move(arg.m_lhs)), apply_eval(eval, std::move(arg.m_rhs)));
                    }
                } // namespace evaluation
            }     // namespace expressions
        }         // namespace cartesian
    }             // namespace stencil
} // namespace gridtools
