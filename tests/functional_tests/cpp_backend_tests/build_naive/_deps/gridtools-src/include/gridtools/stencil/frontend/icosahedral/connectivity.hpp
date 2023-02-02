/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../../common/integral_constant.hpp"
#include "../../../common/tuple.hpp"
#include "../../../meta.hpp"
#include "../../common/extent.hpp"
#include "location_type.hpp"

namespace gridtools {
    namespace stencil {
        namespace icosahedral {
            namespace connectivity_impl_ {
                template <int I, int J, int C>
                using e = tuple<integral_constant<int, I>,
                    integral_constant<int, J>,
                    integral_constant<int, 0>,
                    integral_constant<int, C>>;

                template <class...>
                struct l {
                    using type = l;
                };

                template <class From, class To, int Color>
                struct offsets;

                /**
                 * Following specializations provide all information about the connectivity of the icosahedral/ocahedral
                 grid
                 * While ordering is arbitrary up to some extent, if must respect some rules that user expect, and that
                 conform
                 * part of an API. Rules are the following:
                 *   1. Flow variables on edges by convention are outward on downward cells (color 0) and inward on
                 upward cells
                 * (color 1)
                 *      as depicted below
                 @verbatim
                          ^
                          |                   /\
                     _____|____              /  \
                     \        /             /    \
                      \      /             /      \
                  <----\    /---->        /-->  <--\
                        \  /             /     ^    \
                         \/             /______|_____\
                 @endverbatim
                 *   2. Neighbor edges of a cell must follow the same convention than neighbor cells of a cell. I.e. the
                 following
                 *
                 @verbatim
                         /\
                        1  2
                       /_0__\
                   imposes
                      ____________
                      \    /\    /
                       \1 /  \2 /
                        \/____\/
                         \  0 /
                          \  /
                           \/
                 @endverbatim
                 *
                 *   3. Cell neighbours of an edge, in the order 0 -> 1 follow the direction of the flow (N_t) on edges
                 defined in
                 * 1.
                 *      This fixes the order of cell neighbors of an edge
                 *
                 *   4. Vertex neighbors of an edge, in the order 0 -> 1 defines a vector N_l which is perpendicular to
                 N_t.
                 *      This fixes the order of vertex neighbors of an edge
                 *
                 */
                /*
                 * neighbors order
                 *
                 @verbatim
                   ____________
                   \    /\    /
                    \1 /  \2 /
                     \/____\/
                      \  0 /
                       \  /
                        \/
                 @endverbatim
                 */
                template <>
                struct offsets<cells, cells, 1> : l<e<1, 0, 0>, e<0, 0, 0>, e<0, 1, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim
                         /\
                        /0 \
                       /____\
                      /\    /\
                     /2 \  /1 \
                    /____\/____\
                 @endverbatim
                 */
                template <>
                struct offsets<cells, cells, 0> : l<e<-1, 0, 1>, e<0, 0, 1>, e<0, -1, 1>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                        1____2
                       /\    /\
                      /  \  /  \
                     0____\/____3
                     \    /\    /
                      \  /  \  /
                       \5____4/

                 @endverbatim
                 */
                template <>
                struct offsets<vertices, vertices, 0>
                    : l<e<0, -1, 0>, e<-1, 0, 0>, e<-1, 1, 0>, e<0, 1, 0>, e<1, 0, 0>, e<1, -1, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                       __1___
                      /\    /
                     0  \  2
                    /_3__\/

                 @endverbatim
                 */
                template <>
                struct offsets<edges, edges, 0> : l<e<0, -1, 2>, e<0, 0, 1>, e<0, 0, 2>, e<1, -1, 1>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                     /\
                    0  1
                   /____\
                   \    /
                    3  2
                     \/

                 @endverbatim
                 */
                template <>
                struct offsets<edges, edges, 1> : l<e<-1, 0, 2>, e<-1, 1, 0>, e<0, 0, 2>, e<0, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                   __1___
                   \    /\
                    0  /  2
                     \/_3__\

                 @endverbatim
                 */
                template <>
                struct offsets<edges, edges, 2> : l<e<0, 0, 0>, e<0, 0, 1>, e<0, 1, 0>, e<1, 0, 1>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                      /\
                     1  2
                    /_0__\

                 @endverbatim
                 */
                template <>
                struct offsets<cells, edges, 1> : l<e<1, 0, 1>, e<0, 0, 2>, e<0, 1, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                   __0___
                   \    /
                    2  1
                     \/

                 @endverbatim
                 */
                template <>
                struct offsets<cells, edges, 0> : l<e<0, 0, 1>, e<0, 0, 2>, e<0, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                  1______2
                   \    /
                    \  /
                     \/
                     0

                 @endverbatim
                 */
                template <>
                struct offsets<cells, vertices, 0> : l<e<1, 0, 0>, e<0, 0, 0>, e<0, 1, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                      0
                      /\
                     /  \
                    /____\
                   2      1

                 @endverbatim
                 */
                template <>
                struct offsets<cells, vertices, 1> : l<e<0, 1, 0>, e<1, 1, 0>, e<1, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim
                       ______
                      /\  1 /
                     /0 \  /
                    /____\/

                 @endverbatim
                 */
                template <>
                struct offsets<edges, cells, 0> : l<e<0, -1, 1>, e<0, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                      /\
                     / 0\
                    /____\
                    \    /
                     \1 /
                      \/

                 @endverbatim
                 */
                template <>
                struct offsets<edges, cells, 1> : l<e<-1, 0, 1>, e<0, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim
                   ______
                   \ 1  /\
                    \  / 0\
                     \/____\

                 @endverbatim
                 */
                template <>
                struct offsets<edges, cells, 2> : l<e<0, 0, 1>, e<0, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                      1______
                      /\    /
                     /  \  /
                    /____\/
                          0

                 @endverbatim
                 */
                template <>
                struct offsets<edges, vertices, 0> : l<e<1, 0, 0>, e<0, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                      /\
                     /  \
                   0/____\1
                    \    /
                     \  /
                      \/

                 @endverbatim
                 */
                template <>
                struct offsets<edges, vertices, 1> : l<e<0, 0, 0>, e<0, 1, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim

                   ______0
                   \    /\
                    \  /  \
                     \/____\
                     1

                 @endverbatim
                 */
                template <>
                struct offsets<edges, vertices, 2> : l<e<0, 1, 0>, e<1, 0, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim
                       ______
                      /\ 1  /\
                     /0 \  / 2\
                    /____\/____\
                    \ 5  /\ 3  /
                     \  /4 \  /
                      \/____\/

                 @endverbatim
                 */
                template <>
                struct offsets<vertices, cells, 0>
                    : l<e<-1, -1, 1>, e<-1, 0, 0>, e<-1, 0, 1>, e<0, 0, 0>, e<0, -1, 1>, e<0, -1, 0>> {};

                /*
                 * neighbors order
                 *
                 @verbatim
                       ______
                      /\    /\
                     /  1  2  \
                    /__0_\/__3_\
                    \    /\    /
                     \  5  4  /
                      \/____\/

                 @endverbatim
                 */
                template <>
                struct offsets<vertices, edges, 0>
                    : l<e<0, -1, 1>, e<-1, 0, 0>, e<-1, 0, 2>, e<0, 0, 1>, e<0, 0, 0>, e<0, -1, 2>> {};

                template <class From, class To, int Color>
                using neighbor_offsets = typename offsets<From, To, Color>::type;

                template <class>
                struct offset_to_extent;

                template <int I, int J, int C>
                struct offset_to_extent<e<I, J, C>> {
                    using type = extent<I, I, J, J>;
                };

                template <class From, class To, class = std::make_integer_sequence<int, From::value>>
                struct lazy_neighbors_extent;

                template <class From, class To, int... Colors>
                struct lazy_neighbors_extent<From, To, std::integer_sequence<int, Colors...>> {
                    using type = meta::rename<enclosing_extent,
                        meta::concat<meta::transform<meta::force<offset_to_extent>::apply,
                            neighbor_offsets<From, To, Colors>>...>>;
                };

                template <class From, class To>
                using neighbors_extent = typename lazy_neighbors_extent<From, To>::type;
            } // namespace connectivity_impl_
            using connectivity_impl_::neighbor_offsets;
            using connectivity_impl_::neighbors_extent;

        } // namespace icosahedral
    }     // namespace stencil
} // namespace gridtools
