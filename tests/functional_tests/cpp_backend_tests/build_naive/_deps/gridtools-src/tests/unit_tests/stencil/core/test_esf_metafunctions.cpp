/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/core/compute_extents_metafunctions.hpp>

#include <gridtools/stencil/cartesian.hpp>

using namespace gridtools;
using namespace stencil;
using namespace cartesian;
using namespace core;

struct functor0 {
    typedef in_accessor<0, extent<0, 0, -1, 3, -2, 0>> in0;
    typedef in_accessor<1, extent<-1, 1, 0, 2, -1, 2>> in1;
    typedef in_accessor<2, extent<-3, 3, -1, 2, 0, 1>> in2;
    typedef inout_accessor<3> out;

    typedef make_param_list<in0, in1, in2, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor1 {
    typedef in_accessor<0, extent<0, 1, -1, 2, 0, 0>> in0;
    typedef inout_accessor<1> out;
    typedef in_accessor<2, extent<-3, 0, -3, 0, 0, 2>> in2;
    typedef in_accessor<3, extent<0, 2, 0, 2, -2, 3>> in3;

    typedef make_param_list<in0, out, in2, in3> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor2 {
    typedef in_accessor<0, extent<-3, 3, -1, 0, -2, 1>> in0;
    typedef in_accessor<1, extent<-3, 1, -2, 1, 0, 2>> in1;
    typedef inout_accessor<2> out;

    typedef make_param_list<in0, in1, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor3 {
    typedef in_accessor<0, extent<0, 3, 0, 1, -2, 0>> in0;
    typedef in_accessor<1, extent<-2, 3, 0, 2, -3, 1>> in1;
    typedef inout_accessor<2> out;
    typedef in_accessor<3, extent<-1, 3, -3, 0, -3, 2>> in3;

    typedef make_param_list<in0, in1, out, in3> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor4 {
    typedef in_accessor<0, extent<0, 3, -2, 1, -3, 2>> in0;
    typedef in_accessor<1, extent<-2, 3, 0, 3, -3, 2>> in1;
    typedef in_accessor<2, extent<-1, 1, 0, 3, 0, 3>> in2;
    typedef inout_accessor<3> out;

    typedef make_param_list<in0, in1, in2, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor5 {
    typedef in_accessor<0, extent<-3, 1, -1, 2, -1, 1>> in0;
    typedef in_accessor<1, extent<0, 1, -2, 2, 0, 3>> in1;
    typedef in_accessor<2, extent<0, 2, 0, 3, -1, 2>> in2;
    typedef inout_accessor<3> out;

    typedef make_param_list<in0, in1, in2, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor6 {
    typedef inout_accessor<0> out;
    typedef in_accessor<1, extent<0, 3, -3, 2, 0, 0>> in1;
    typedef in_accessor<2, extent<-3, 2, 0, 2, -1, 2>> in2;
    typedef in_accessor<3, extent<-1, 0, -1, 0, -1, 3>> in3;

    typedef make_param_list<out, in1, in2, in3> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

template <int>
struct p {};

typedef p<0> o0;
typedef p<1> o1;
typedef p<2> o2;
typedef p<3> o3;
typedef p<4> o4;
typedef p<5> o5;
typedef p<6> o6;
typedef p<7> in0;
typedef p<8> in1;
typedef p<9> in2;
typedef p<10> in3;

using mss_t = meta::first<decltype(execute_parallel()
                                       .stage(functor0(), in0(), in1(), in2(), o0())
                                       .stage(functor1(), in3(), o1(), in0(), o0())
                                       .stage(functor2(), o0(), o1(), o2())
                                       .stage(functor3(), in1(), in2(), o3(), o2())
                                       .stage(functor4(), o0(), o1(), o3(), o4())
                                       .stage(functor5(), in3(), o4(), in0(), o5())
                                       .stage(functor6(), o6(), o5(), in1(), in2()))>;

template <class Arg, int_t... ExpectedExtentValues>
using testee = std::is_same<lookup_extent_map<get_extent_map_from_mss<mss_t>, Arg>, extent<ExpectedExtentValues...>>;

static_assert(testee<o0, -5, 11, -10, 10, -5, 13>::value);
static_assert(testee<o1, -5, 9, -10, 8, -3, 10>::value);
static_assert(testee<o2, -2, 8, -8, 7, -3, 8>::value);
static_assert(testee<o3, -1, 5, -5, 7, 0, 6>::value);
static_assert(testee<o4, 0, 4, -5, 4, 0, 3>::value);
static_assert(testee<o5, 0, 3, -3, 2, 0, 0>::value);
static_assert(testee<o6, 0, 0, 0, 0, 0, 0>::value);
static_assert(testee<in0, -8, 11, -13, 13, -7, 13>::value);
static_assert(testee<in1, -6, 12, -10, 12, -6, 15>::value);
static_assert(testee<in2, -8, 14, -11, 12, -5, 14>::value);
static_assert(testee<in3, -5, 10, -11, 10, -3, 10>::value);
