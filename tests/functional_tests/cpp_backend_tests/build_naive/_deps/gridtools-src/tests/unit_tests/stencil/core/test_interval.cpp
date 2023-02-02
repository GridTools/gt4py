/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/core/interval.hpp>

using namespace gridtools;
using namespace stencil;
using namespace core;

constexpr int level_offset_limit = 3;

template <uint_t Splitter, int_t Offset>
using level_t = level<Splitter, Offset, level_offset_limit>;

using my_interval = interval<level_t<0, -1>, level_t<1, -1>>;

template <class T, uint_t FromSplitter, int_t FromOffset, uint_t ToSplitter, int_t ToOffset>
constexpr bool testee = std::is_same_v<T, interval<level_t<FromSplitter, FromOffset>, level_t<ToSplitter, ToOffset>>>;

static_assert(testee<my_interval::modify<-1, 0>, 0, -2, 1, -1>);
static_assert(testee<my_interval::modify<1, 1>, 0, 1, 1, 1>);
static_assert(testee<my_interval::modify<-2, 0>, 0, -3, 1, -1>);
static_assert(testee<my_interval::modify<2, 2>, 0, 2, 1, 2>);
