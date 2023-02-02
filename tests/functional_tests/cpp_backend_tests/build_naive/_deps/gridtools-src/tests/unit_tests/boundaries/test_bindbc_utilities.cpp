/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/boundaries/bound_bc.hpp>

#include <utility>

#include <gtest/gtest.h>

#include <gridtools/boundaries/zero.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/cpu_kfirst.hpp>

using namespace std::placeholders;
namespace gt = gridtools;
namespace bd = gt::boundaries;

TEST(DistributedBoundaries, SelectElement) {
    auto all = std::make_tuple(1, _1, 3, _2);
    auto sub = std::make_tuple(2, 4);

    EXPECT_EQ(bd::_impl::select_element<0>(sub, all, bd::_impl::NotPlc{}), 1);
    EXPECT_EQ(bd::_impl::select_element<1>(sub, all, bd::_impl::Plc{}), 2);
    EXPECT_EQ(bd::_impl::select_element<2>(sub, all, bd::_impl::NotPlc{}), 3);
    EXPECT_EQ(bd::_impl::select_element<3>(sub, all, bd::_impl::Plc{}), 4);
}

namespace collect_indices {
    template <class Tuple, size_t... Is>
    constexpr bool testee = std::is_same_v<
        typename bd::_impl::comm_indices<std::tuple<>>::collect_indices<0, std::index_sequence<>, Tuple>::type,
        std::index_sequence<Is...>>;

    static_assert(testee<std::tuple<int, int>, 0, 1>);
    static_assert(testee<std::tuple<int, decltype(_1), int, decltype(_2)>, 0, 2>);
    static_assert(testee<std::tuple<decltype(_1), decltype(_2)>>);
} // namespace collect_indices

TEST(DistributedBoundaries, RestTuple) {
    EXPECT_EQ(bd::_impl::rest_tuple(std::make_tuple(), std::make_index_sequence<0>{}), std::make_tuple());
    EXPECT_EQ(bd::_impl::rest_tuple(std::make_tuple(1), std::make_index_sequence<0>{}), std::make_tuple());
    EXPECT_EQ(bd::_impl::rest_tuple(std::make_tuple(1, 2), std::make_index_sequence<1>{}), std::make_tuple(2));
}

static_assert(!bd::_impl::contains_placeholders<decltype(std::make_tuple(3, 4, 5))>::value);
static_assert(!bd::_impl::contains_placeholders<decltype(std::make_tuple())>::value);
static_assert(bd::_impl::contains_placeholders<decltype(std::make_tuple(3, 4, _1))>::value);
static_assert(bd::_impl::contains_placeholders<decltype(std::make_tuple(3, _2, 5))>::value);

TEST(DistributedBoundaries, BoundBC) {
    const auto builder = gt::storage::builder<gt::storage::cpu_kfirst>.type<double>().dimensions(3, 3, 3);

    auto a = builder();
    auto b = builder();
    auto c = builder();

    using ds = decltype(a);

    bd::bound_bc<bd::zero_boundary, std::tuple<ds, ds, ds>, std::index_sequence<1>> bbc{
        bd::zero_boundary{}, std::make_tuple(a, b, c)};

    auto x = bbc.stores();

    EXPECT_EQ(a, std::get<0>(x));
    EXPECT_EQ(b, std::get<1>(x));
    EXPECT_EQ(c, std::get<2>(x));

    auto y = bbc.exc_stores();

    static_assert(std::tuple_size_v<decltype(y)> == 1);

    EXPECT_EQ(b, std::get<0>(y));
}
