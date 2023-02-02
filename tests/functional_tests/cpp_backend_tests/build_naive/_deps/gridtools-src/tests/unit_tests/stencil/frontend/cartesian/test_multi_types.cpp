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
#include <gridtools/stencil/naive.hpp>

#define GT_STENCIL_NAIVE
#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace {
                struct type1 {
                    int i, j, k;

                    GT_FUNCTION type1() : i(0), j(0), k(0) {}
                    GT_FUNCTION type1(int i, int j, int k) : i(i), j(j), k(k) {}
                };

                struct type4 {
                    float x, y, z;

                    GT_FUNCTION type4() : x(0.), y(0.), z(0.) {}
                    GT_FUNCTION type4(double i, double j, double k) : x(i), y(j), z(k) {}

                    GT_FUNCTION type4 &operator=(type1 const &a) {
                        x = a.i;
                        y = a.j;
                        z = a.k;
                        return *this;
                    }
                };

                struct type2 {
                    double xy;
                    GT_FUNCTION type2 &operator=(type4 x) {
                        xy = x.x + x.y;
                        return *this;
                    }
                    friend std::ostream &operator<<(std::ostream &strm, type2 obj) {
                        return strm << "{ xy: " << obj.xy << " }";
                    }
                    friend GT_FUNCTION bool operator==(type2 lhs, type2 rhs) { return lhs.xy == rhs.xy; }
                };

                struct type3 {
                    double yz;

                    GT_FUNCTION type3 &operator=(type4 x) {
                        yz = x.y + x.z;
                        return *this;
                    }
                    friend std::ostream &operator<<(std::ostream &strm, type3 obj) {
                        return strm << "{ yz: " << obj.yz << " }";
                    }
                    friend GT_FUNCTION bool operator==(type3 lhs, type3 rhs) { return lhs.yz == rhs.yz; }
                };

                GT_FUNCTION type4 operator+(type4 a, type1 b) {
                    return {a.x + double(b.i), a.y + double(b.j), a.z + double(b.k)};
                }

                GT_FUNCTION type4 operator-(type4 a, type1 b) {
                    return {a.x - double(b.i), a.y - double(b.j), a.z - double(b.k)};
                }

                GT_FUNCTION type4 operator+(type1 a, type4 b) {
                    return {a.i + double(b.x), a.j + double(b.y), a.k + double(b.z)};
                }

                GT_FUNCTION type4 operator-(type1 a, type4 b) {
                    return {a.i - double(b.x), a.j - double(b.y), a.k - double(b.z)};
                }

                GT_FUNCTION type4 operator+(type1 a, type1 b) {
                    return {a.i + double(b.i), a.j + double(b.j), a.k + double(b.k)};
                }

                GT_FUNCTION type4 operator-(type1 a, type1 b) {
                    return {a.i - double(b.i), a.j - double(b.j), a.k - double(b.k)};
                }

                struct function0 {
                    using in = in_accessor<0>;
                    using out = inout_accessor<1>;

                    using param_list = make_param_list<in, out>;

                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval) {
                        eval(out()).i = eval(in()).i + 1;
                        eval(out()).j = eval(in()).j + 1;
                        eval(out()).k = eval(in()).k + 1;
                    }
                };

                enum class call_type { function, procedure };

                template <call_type Type, class Eval, class T, std::enable_if_t<Type == call_type::function, int> = 0>
                GT_FUNCTION auto call_function0(Eval &eval, T obj) {
                    return call<function0>::with(eval, obj);
                }

                template <call_type Type, class Eval, class T, std::enable_if_t<Type == call_type::procedure, int> = 0>
                GT_FUNCTION type1 call_function0(Eval &eval, T obj) {
                    type1 res;
                    call_proc<function0>::with(eval, obj, res);
                    return res;
                }

                template <call_type CallType>
                struct function1 {
                    using out = inout_accessor<0>;
                    using in = in_accessor<1>;

                    using param_list = make_param_list<out, in>;

                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval) {
                        eval(out()) = call_function0<CallType>(eval, in());
                    }
                };

                struct function2 {
                    using out = inout_accessor<0>;
                    using in = in_accessor<1>;
                    using temp = in_accessor<2>;

                    using param_list = make_param_list<out, in, temp>;

                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval) {
                        eval(out()) = eval(temp()) + eval(in());
                    }
                };

                struct function3 {
                    using out = inout_accessor<0>;
                    using temp = in_accessor<1>;
                    using in = in_accessor<2>;

                    using param_list = make_param_list<out, temp, in>;

                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval) {
                        eval(out()) = eval(temp()) - eval(in());
                    }
                };

                template <call_type CallType>
                void do_test() {
                    using env = test_environment<>::apply<naive, double, inlined_params<4, 5, 6>>;
                    auto in = [](int i, int j, int k) -> type1 { return {i, j, k}; };
                    auto field2 = env::make_storage<type2>();
                    auto field3 = env::make_storage<type3>();

                    using fun1 = function1<CallType>;

                    run(
                        [](auto field1, auto field2, auto field3) {
                            GT_DECLARE_TMP(type1, temp);
                            return multi_pass(
                                execute_forward().stage(fun1(), temp, field1).stage(function2(), field2, field1, temp),
                                execute_backward()
                                    .stage(fun1(), temp, field1)
                                    .stage(function3(), field3, temp, field1));
                        },
                        naive(),
                        env::make_grid(),
                        env::make_storage<type1>(in),
                        field2,
                        field3);

                    env::verify(
                        [&in](int i, int j, int k) {
                            auto f1 = in(i, j, k);
                            type2 res = {2. * (f1.i + f1.j + 1)};
                            return res;
                        },
                        field2);

                    env::verify(type3{2}, field3);
                }

                TEST(multitypes, function) { do_test<call_type::function>(); }

                TEST(multitypes, procedure) { do_test<call_type::procedure>(); }

            } // namespace
        }     // namespace cartesian
    }         // namespace stencil
} // namespace gridtools
