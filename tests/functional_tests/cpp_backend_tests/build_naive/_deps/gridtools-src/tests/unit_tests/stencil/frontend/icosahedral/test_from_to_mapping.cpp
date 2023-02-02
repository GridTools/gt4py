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

#include <gridtools/meta.hpp>
#include <gridtools/stencil/icosahedral.hpp>

using namespace gridtools;
using namespace stencil;
using namespace icosahedral;

template <class From, class To, int Color>
constexpr bool testee = meta::first<meta::first<neighbor_offsets<From, To, Color>>>::value < 10;

// From Cells to XXX
static_assert(testee<cells, cells, 0>);
static_assert(testee<cells, cells, 1>);
static_assert(testee<cells, edges, 0>);
static_assert(testee<cells, edges, 1>);
static_assert(testee<cells, vertices, 0>);
static_assert(testee<cells, vertices, 1>);

// From Edges to XXX
static_assert(testee<edges, cells, 0>);
static_assert(testee<edges, cells, 1>);
static_assert(testee<edges, cells, 2>);
static_assert(testee<edges, edges, 0>);
static_assert(testee<edges, edges, 1>);
static_assert(testee<edges, edges, 2>);
static_assert(testee<edges, vertices, 0>);
static_assert(testee<edges, vertices, 1>);
static_assert(testee<edges, vertices, 2>);

// From Vertices to XXX
static_assert(testee<vertices, cells, 0>);
static_assert(testee<vertices, edges, 0>);
static_assert(testee<vertices, vertices, 0>);

static_assert(std::is_same_v<neighbors_extent<cells, cells>, extent<-1, 1, -1, 1>>);
static_assert(std::is_same_v<neighbors_extent<cells, edges>, extent<0, 1, 0, 1>>);
static_assert(std::is_same_v<neighbors_extent<cells, vertices>, extent<0, 1, 0, 1>>);

static_assert(std::is_same_v<neighbors_extent<edges, cells>, extent<-1, 0, -1, 0>>);
static_assert(std::is_same_v<neighbors_extent<edges, edges>, extent<-1, 1, -1, 1>>);
static_assert(std::is_same_v<neighbors_extent<edges, vertices>, extent<0, 1, 0, 1>>);

static_assert(std::is_same_v<neighbors_extent<vertices, cells>, extent<-1, 0, -1, 0>>);
static_assert(std::is_same_v<neighbors_extent<vertices, edges>, extent<-1, 0, -1, 0>>);
static_assert(std::is_same_v<neighbors_extent<vertices, vertices>, extent<-1, 1, -1, 1>>);
