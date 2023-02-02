/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <gridtools/stencil/cartesian.hpp>

namespace gridtools {
    namespace stencil {
        namespace cartesian {

            using namespace expressions;

            /*
             * Mocking accessor and iterate domain
             */
            template <class T>
            struct accessor_mock {
                T value;
            };

            template <class T>
            struct is_accessor<accessor_mock<T>> : std::true_type {};

            namespace {

                struct iterate_domain_mock {
                    // trivial evaluation of the accessor_mock
                    template <typename Accessor, typename std::enable_if<is_accessor<Accessor>::value, int>::type = 0>
                    GT_FUNCTION auto operator()(Accessor const &val) const {
                        return val.value;
                    }

                    // copy of the iterate_domain for expr
                    template <class Op, class... Args>
                    GT_FUNCTION auto operator()(expr<Op, Args...> const &arg) const {
                        return expressions::evaluation::value(*this, arg);
                    }
                };

                using val = accessor_mock<float>;

#define EXPRESSION_TEST(name, expr)                             \
    GT_FUNCTION float test_##name() {                           \
        auto result = iterate_domain_mock()(expr);              \
        static_assert(std::is_same_v<decltype(result), float>); \
        return result;                                          \
    }

                EXPRESSION_TEST(add_accessors, val{1} + val{2})
                EXPRESSION_TEST(sub_accessors, val{1} - val{2})
                EXPRESSION_TEST(negate_accessors, -val{1})
                EXPRESSION_TEST(plus_sign_accessors, +val{1})
                EXPRESSION_TEST(with_parenthesis_accessors, (val{1} + val{2}) * (val{1} - val{2}))

                /*
                 * User API tests
                 */
                TEST(test_expressions, add_accessors) {
                    EXPECT_FLOAT_EQ(test_add_accessors(), 3);
                    EXPECT_FLOAT_EQ(test_sub_accessors(), -1);
                    EXPECT_FLOAT_EQ(test_negate_accessors(), -1);
                    EXPECT_FLOAT_EQ(test_plus_sign_accessors(), 1);
                    EXPECT_FLOAT_EQ(test_with_parenthesis_accessors(), -3);
                }

                /*
                 * Library tests illustrating how expressions are evaluated (implementation details!)
                 */
                TEST(test_expressions, expr_plus_by_ctor) {
                    accessor_mock<double> a{1};
                    accessor_mock<double> b{2};

                    auto add = make_expr(plus_f{}, a, b);

                    iterate_domain_mock eval;

                    auto result = evaluation::value(eval, add);

                    ASSERT_DOUBLE_EQ(result, 3);
                }

                TEST(expression_unit_test, expr_plus_by_operator_overload) {
                    iterate_domain_mock iterate;

                    auto add = val{1} + val{2};

                    auto result = evaluation::value(iterate, add);

                    ASSERT_DOUBLE_EQ(result, 3);
                }
            } // namespace
        }     // namespace cartesian
    }         // namespace stencil
} // namespace gridtools
