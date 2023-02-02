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
 * Compile-time test to ensure that types are correct in all call_proc stages
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

struct in_tag {};
struct out_tag {};
struct local_tag {};

struct triple_nesting_with_type_switching_third_stage {
    typedef inout_accessor<0> out;
    typedef in_accessor<1> local;
    typedef make_param_list<out, local> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        static_assert(check_acceessor<Evaluation, out, out_tag>::value);
        static_assert(check_acceessor<Evaluation, local, local_tag>::value);
    }
};

struct triple_nesting_with_type_switching_second_stage {
    typedef in_accessor<0> in;
    typedef inout_accessor<1> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        static_assert(check_acceessor<Evaluation, out, out_tag>::value);
        static_assert(check_acceessor<Evaluation, in, in_tag>::value);

        local_tag local;
        call_proc<triple_nesting_with_type_switching_third_stage>::with(eval, out(), local);
    }
};

struct triple_nesting_with_type_switching_first_stage {
    typedef inout_accessor<0> out;
    typedef in_accessor<1> in;
    typedef make_param_list<out, in> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        static_assert(check_acceessor<Evaluation, out, out_tag>::value);
        static_assert(check_acceessor<Evaluation, in, in_tag>::value);

        call_proc<triple_nesting_with_type_switching_second_stage>::with(eval, in(), out());
    }
};

void dummy() { do_test<triple_nesting_with_type_switching_first_stage, out_tag, in_tag>(); }
