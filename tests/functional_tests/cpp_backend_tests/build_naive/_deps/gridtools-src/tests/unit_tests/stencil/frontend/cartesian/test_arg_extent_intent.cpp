/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/cartesian.hpp>

using namespace gridtools;
using namespace stencil;
using namespace cartesian;

struct stage1 {
    using in1 = in_accessor<0, extent<0, 1, -1, 0, 0, 1>>;
    using in2 = in_accessor<1, extent<0, 1, -1, 0, -1, 1>>;
    using out = inout_accessor<2>;
    using param_list = make_param_list<in1, in2, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&) {}
};

struct stage2 {
    using in1 = in_accessor<0, extent<-1, 0, 0, 1, -1, 0>>;
    using in2 = in_accessor<1, extent<-1, 1, -1, 0, -1, 1>>;
    using out = inout_accessor<2>;
    using param_list = make_param_list<in1, in2, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&) {}
};

template <class Arg, int... Is, class Spec>
constexpr bool test_extent(Spec spec) {
    return std::is_same_v<decltype(get_arg_extent(spec, Arg())), extent<Is...>>;
}

template <class Plh, class Spec>
constexpr bool test_intent(Spec spec, intent expected) {
    return decltype(get_arg_intent(spec, Plh()))::value == expected;
}

struct a {};
struct b {};
struct c {};
struct d {};
struct e {};
struct f {};

constexpr auto mss0 = execute_parallel().stage(stage1(), a(), b(), c());

static_assert(test_extent<a, 0, 1, -1, 0, 0, 1>(mss0));
static_assert(test_extent<b, 0, 1, -1, 0, -1, 1>(mss0));
static_assert(test_extent<c>(mss0));

static_assert(test_intent<a>(mss0, intent::in));
static_assert(test_intent<b>(mss0, intent::in));
static_assert(test_intent<c>(mss0, intent::inout));

constexpr auto mss1 = execute_parallel().stage(stage1(), a(), b(), c()).stage(stage2(), a(), c(), d());

static_assert(test_extent<a, -1, 2, -2, 1, -1, 2>(mss1));
static_assert(test_extent<b, -1, 2, -2, 0, -2, 2>(mss1));
static_assert(test_extent<c, -1, 1, -1, 0, -1, 1>(mss1));
static_assert(test_extent<d>(mss1));

static_assert(test_intent<a>(mss1, intent::in));
static_assert(test_intent<b>(mss1, intent::in));
static_assert(test_intent<c>(mss1, intent::inout));
static_assert(test_intent<d>(mss1, intent::inout));

constexpr auto mss2 = execute_parallel()
                          .stage(stage1(), a(), b(), c())
                          .stage(stage1(), a(), c(), d())
                          .stage(stage2(), b(), c(), e())
                          .stage(stage2(), d(), e(), f());

// after last stage:
//   f:  {0, 0, 0, 0, 0, 0}
//   e: {-1, 1, -1, 0, -1, 1}
//   d: {-1, 0, 0, 1, -1, 0}
//
// after third stage:
//   f:  {0, 0, 0, 0, 0, 0}
//   e: {-1, 1, -1, 0, -1, 1}
//   d: {-1, 0, 0, 1, -1, 0}
//   c: {-2, 2, -2, 0, -2, 2}
//   b:  {-2, 1, -1, 1, -2, 1}
//
// after second stage:
//   f:  {0, 0, 0, 0, 0, 0}
//   e: {-1, 1, -1, 0, -1, 1}
//   d: {-1, 0, 0, 1, -1, 0}
//   c {-2, 2, -2, 1, -2, 2}
//   b:  {-2, 1, -1, 1, -2, 1}
//   a:  {-1, 1, -1, 1, -1, 1}
//
// after first stage:
//   f:  {0, 0, 0, 0, 0, 0}
//   e: {-1, 1, -1, 0, -1, 1}
//   d: {-1, 0, 0, 1, -1, 0}
//   c: {-2, 2, -2, 1, -2, 2}
//   b:  {-2, 3, -3, 1, -3, 3}
//   a:  {-2, 3, -3, 1, -2, 3}

static_assert(test_extent<a, -2, 3, -3, 1, -2, 3>(mss2));
static_assert(test_extent<b, -2, 3, -3, 1, -3, 3>(mss2));
static_assert(test_extent<c, -2, 2, -2, 1, -2, 2>(mss2));
static_assert(test_extent<d, -1, 0, 0, 1, -1, 0>(mss2));
static_assert(test_extent<e, -1, 1, -1, 0, -1, 1>(mss2));
static_assert(test_extent<f>(mss2));

static_assert(test_intent<a>(mss2, intent::in));
static_assert(test_intent<b>(mss2, intent::in));
static_assert(test_intent<c>(mss2, intent::inout));
static_assert(test_intent<d>(mss2, intent::inout));
static_assert(test_intent<e>(mss2, intent::inout));
static_assert(test_intent<f>(mss2, intent::inout));
