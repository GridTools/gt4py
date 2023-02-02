/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <type_traits>

#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/naive.hpp>

/**
 * Compile-time test to ensure that types are correct in all call stages
 */

using namespace gridtools;
using namespace stencil;
using namespace cartesian;

template <class T>
T storage[1];

template <class Fun, class... Args>
void do_test() {
    run_single_stage(Fun(), naive(), make_grid(1, 1, 1), storage<Args>...);
}

template <class Eval, class Acc, class Expected>
using check_acceessor = std::is_same<std::decay_t<decltype(std::declval<Eval &>()(Acc()))>, Expected>;

struct in1_tag {};
struct in2_tag {};
struct out_tag {};
struct forced_tag {};
struct local_tag {};

struct simple_callee_with_forced_return_type {
    typedef in_accessor<0> in;
    typedef inout_accessor<1> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        static_assert(check_acceessor<Evaluation, out, forced_tag>::value);
        static_assert(check_acceessor<Evaluation, in, in1_tag>::value);
    }
};

struct simple_caller_with_forced_return_type {
    typedef in_accessor<0> in;
    typedef inout_accessor<1> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        auto result = call<simple_callee_with_forced_return_type>::return_type<forced_tag>::with(eval, in{});
        static_assert(std::is_same_v<decltype(result), forced_tag>);
    }

    void dummy() { do_test<simple_caller_with_forced_return_type, in1_tag, out_tag>(); }
};

struct simple_callee_with_deduced_return_type {
    typedef in_accessor<0> in;
    typedef inout_accessor<1> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        static_assert(check_acceessor<Evaluation, out, in1_tag>::value);
        static_assert(check_acceessor<Evaluation, in, in1_tag>::value);
    }
};

struct simple_caller_with_deduced_return_type {
    typedef in_accessor<0> in;
    typedef inout_accessor<1> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        auto result = call<simple_callee_with_deduced_return_type>::with(eval, in{});
        static_assert(std::is_same_v<decltype(result), in1_tag>);
    }
    void dummy() { do_test<simple_caller_with_deduced_return_type, in1_tag, out_tag>(); }
};

struct triple_nesting_with_type_switching_third_stage {
    typedef in_accessor<0> in2;
    typedef in_accessor<1> local;
    typedef inout_accessor<2> out;
    typedef in_accessor<3> in1;
    typedef make_param_list<in2, local, out, in1> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        // the new convention is that the return type (here "out) is deduced from the first argument in the call
        static_assert(check_acceessor<Evaluation, out, in2_tag>::value);
        static_assert(check_acceessor<Evaluation, in1, in1_tag>::value);
        static_assert(check_acceessor<Evaluation, in2, in2_tag>::value);
        static_assert(check_acceessor<Evaluation, local, local_tag>::value);
    }
};

struct triple_nesting_with_type_switching_second_stage {
    typedef in_accessor<0> in1;
    typedef inout_accessor<1> out;
    typedef in_accessor<2> in2;
    typedef make_param_list<in1, out, in2> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        using out_type = std::decay_t<decltype(eval(out{}))>;
        // the expected type differs here in "call" vs "call_proc"
        static_assert(check_acceessor<Evaluation, out, in1_tag>::value);
        static_assert(check_acceessor<Evaluation, in1, in1_tag>::value);
        static_assert(check_acceessor<Evaluation, in2, in2_tag>::value);

        local_tag local;

        auto result = call<triple_nesting_with_type_switching_third_stage>::with(eval, in2(), local, in1());
        static_assert(std::is_same_v<decltype(result), in2_tag>);
    }
};

struct triple_nesting_with_type_switching_first_stage {
    typedef in_accessor<0> in1;
    typedef inout_accessor<1> out;
    typedef in_accessor<2> in2;
    typedef make_param_list<in1, out, in2> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        static_assert(check_acceessor<Evaluation, out, out_tag>::value);
        static_assert(check_acceessor<Evaluation, in1, in1_tag>::value);
        static_assert(check_acceessor<Evaluation, in2, in2_tag>::value);

        auto result = call<triple_nesting_with_type_switching_second_stage>::with(eval, in1(), in2());
        static_assert(std::is_same_v<decltype(result), in1_tag>);
    }
    void dummy() { do_test<triple_nesting_with_type_switching_first_stage, in1_tag, out_tag, in2_tag>(); }
};

struct triple_nesting_with_type_switching_and_call_proc_second_stage {
    typedef in_accessor<0> in1;
    typedef inout_accessor<1> out;
    typedef in_accessor<2> in2;
    typedef make_param_list<in1, out, in2> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        // in contrast to the example where this is stage is called from "call" (not "call_proc")
        // the type here is different!
        static_assert(check_acceessor<Evaluation, out, out_tag>::value);
        static_assert(check_acceessor<Evaluation, in1, in1_tag>::value);
        static_assert(check_acceessor<Evaluation, in2, in2_tag>::value);

        local_tag local;

        auto result = call<triple_nesting_with_type_switching_third_stage>::with(eval, in2(), local, in1());
        static_assert(std::is_same_v<decltype(result), in2_tag>);
    }
};

struct triple_nesting_with_type_switching_and_call_proc_first_stage {
    typedef in_accessor<0> in1;
    typedef inout_accessor<1> out;
    typedef in_accessor<2> in2;
    typedef make_param_list<in1, out, in2> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        static_assert(check_acceessor<Evaluation, out, out_tag>::value);
        static_assert(check_acceessor<Evaluation, in1, in1_tag>::value);
        static_assert(check_acceessor<Evaluation, in2, in2_tag>::value);

        call_proc<triple_nesting_with_type_switching_and_call_proc_second_stage>::with(eval, in1(), out(), in2());
    }
    void dummy() { do_test<triple_nesting_with_type_switching_and_call_proc_first_stage, in1_tag, out_tag, in2_tag>(); }
};
