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

#include "test_call_interfaces.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace {
                struct call_proc_interface : public base_fixture {
                    using storage_type = typename env_t::storage_type;
                    storage_type in = env_t::make_storage(input);
                    storage_type out1 = env_t::make_storage();
                    storage_type out2 = env_t::make_storage();

                    void verify(storage_type actual, fun_t expected = {}) {
                        if (!expected)
                            expected = input;
                        env_t::verify(env_t::make_storage(expected), actual);
                    }
                };

                struct copy_twice_functor {
                    typedef in_accessor<0> in;
                    typedef inout_accessor<1> out1;
                    typedef inout_accessor<2> out2;
                    typedef make_param_list<in, out1, out2> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        eval(out1()) = eval(in());
                        eval(out2()) = eval(in());
                    }
                };

                struct call_copy_functor_with_expression {
                    typedef in_accessor<0> in;
                    typedef inout_accessor<1> out;
                    typedef make_param_list<in, out> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<copy_functor_with_expression, x_interval>::with(eval, in(), out());
                    }
                };

                TEST_F(call_proc_interface, call_to_copy_functor_with_expression) {
                    run_computation<call_copy_functor_with_expression>(in, out1);
                    verify(out1);
                }

                struct call_copy_twice_functor {
                    typedef in_accessor<0> in;
                    typedef inout_accessor<1> out1;
                    typedef inout_accessor<2> out2;
                    typedef make_param_list<in, out1, out2> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<copy_twice_functor, x_interval>::with(eval, in(), out1(), out2());
                    }
                };

                TEST_F(call_proc_interface, call_to_copy_twice_functor) {
                    run_computation<call_copy_twice_functor>(in, out1, out2);
                    verify(out1);
                    verify(out2);
                }

                struct call_with_offsets_copy_twice_functor {
                    typedef in_accessor<0, extent<0, 1, 0, 1>> in;
                    typedef inout_accessor<1> out1;
                    typedef inout_accessor<2> out2;
                    typedef make_param_list<in, out1, out2> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<copy_twice_functor, x_interval>::with(eval, in(1, 1, 0), out1(), out2());
                    }
                };

                TEST_F(call_proc_interface, call_with_offsets_to_copy_twice_functor) {
                    run_computation<call_with_offsets_copy_twice_functor>(in, out1, out2);
                    verify(out1, shifted);
                    verify(out2, shifted);
                }

                struct call_proc_copy_functor_default_interval {
                    typedef in_accessor<0> in;
                    typedef inout_accessor<1> out;
                    typedef make_param_list<in, out> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval) {
                        call_proc<copy_functor_default_interval>::with(eval, in(), out());
                    }
                };

                TEST_F(call_proc_interface, call_to_copy_functor_default_interval) {
                    run_computation<call_proc_copy_functor_default_interval>(in, out1);
                    verify(out1);
                }

                struct call_proc_copy_functor_default_interval_with_offset_in_k {
                    typedef in_accessor<0, extent<0, 0, 0, 0, 0, 1>> in;
                    typedef inout_accessor<1, extent<0, 0, 0, 0, 0, 1>> out;
                    typedef make_param_list<in, out> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<copy_functor_default_interval>::at<0, 0, -1>::with(eval, in(0, 0, 1), out(0, 0, 1));
                    }
                };

                TEST_F(call_proc_interface, call_to_copy_functor_default_interval_with_offset_in_k) {
                    run_computation<call_proc_copy_functor_default_interval_with_offset_in_k>(in, out1);
                    verify(out1);
                }

                struct call_call_copy_twice_functor {
                    typedef in_accessor<0> in;
                    typedef inout_accessor<1> out1;
                    typedef inout_accessor<2> out2;
                    typedef make_param_list<in, out1, out2> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<call_copy_twice_functor, x_interval>::with(eval, in(), out1(), out2());
                    }
                };

                TEST_F(call_proc_interface, call_to_call_to_copy_twice_functor) {
                    run_computation<call_call_copy_twice_functor>(in, out1, out2);
                    verify(out1);
                    verify(out2);
                }

                struct call_with_offsets_call_copy_twice_functor {
                    typedef in_accessor<0, extent<0, 1, 0, 1>> in;
                    typedef inout_accessor<1> out1;
                    typedef inout_accessor<2> out2;
                    typedef make_param_list<in, out1, out2> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<call_copy_twice_functor, x_interval>::with(eval, in(1, 1, 0), out1(), out2());
                    }
                };

                TEST_F(call_proc_interface, call_with_offsets_to_call_to_copy_twice_functor) {
                    run_computation<call_with_offsets_call_copy_twice_functor>(in, out1, out2);
                    verify(out1, shifted);
                    verify(out2, shifted);
                }

                struct call_with_offsets_call_with_offsets_copy_twice_functor {
                    typedef in_accessor<0, extent<-1, 0, -1, 0>> in;
                    typedef inout_accessor<1> out1;
                    typedef inout_accessor<2> out2;
                    typedef make_param_list<in, out1, out2> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<call_with_offsets_copy_twice_functor, x_interval>::with(
                            eval, in(-1, -1, 0), out1(), out2());
                    }
                };

                TEST_F(call_proc_interface, call_with_offsets_to_call_with_offsets_to_copy_twice_functor) {
                    run_computation<call_with_offsets_call_with_offsets_copy_twice_functor>(in, out1, out2);
                    verify(out1);
                    verify(out2);
                }

                struct call_with_local_variable {
                    typedef in_accessor<0> in;
                    typedef inout_accessor<1> out1;
                    typedef inout_accessor<2> out2;
                    typedef make_param_list<in, out1, out2> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        double local_in = 1;
                        double local_out = -1;

                        call_proc<copy_functor, x_interval>::with(eval, local_in, local_out);

                        if (local_out > 0.) {
                            eval(out1()) = eval(in());
                        }
                    }
                };

                TEST_F(call_proc_interface, call_using_local_variables) {
                    run_computation<call_with_local_variable>(in, out1, out2);
                    verify(out1);
                }

                struct functor_where_index_of_accessor_is_shifted_inner {
                    typedef inout_accessor<0> out;
                    typedef make_param_list<out> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        eval(out()) = 1.;
                    }
                };

                struct functor_where_index_of_accessor_is_shifted {
                    typedef inout_accessor<0> local_out;
                    typedef inout_accessor<1> out;
                    typedef make_param_list<local_out, out> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        call_proc<functor_where_index_of_accessor_is_shifted_inner, x_interval>::with(eval, out());
                    }
                };

                struct call_with_nested_calls_and_shifted_accessor_index {
                    typedef inout_accessor<0> out;
                    typedef make_param_list<out> param_list;
                    template <typename Evaluation>
                    GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
                        double local_out;
                        call_proc<functor_where_index_of_accessor_is_shifted, x_interval>::with(eval, local_out, out());
                    }
                };

                TEST_F(call_proc_interface, call_using_local_variables_and_nested_call) {
                    run_computation<call_with_nested_calls_and_shifted_accessor_index>(out1);
                    verify(out1, [](int, int, int) { return 1; });
                }
            } // namespace
        }     // namespace cartesian
    }         // namespace stencil
} // namespace gridtools
