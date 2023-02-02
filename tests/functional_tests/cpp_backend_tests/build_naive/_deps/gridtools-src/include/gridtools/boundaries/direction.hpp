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

/**
@file
@brief definition of direction in a 3D cartesian grid
 */
namespace gridtools {
    namespace boundaries {

        /** \ingroup Boundary-Conditions
         * @{
         */

        /**
           @brief Enum defining the directions in a discrete Cartesian grid
         */
        enum sign { any_ = -2, minus_ = -1, zero_, plus_ };

        /** \ingroup Boundary-Conditions
           @brief Class defining a direction in a cartesian 3D grid.

           The directions correspond to the following:
           - all the three template parameters are either plus or minus: identifies a node on the cell
           \verbatim
           e.g. direction<minus_, plus_, minus_> corresponds to:
             .____.
            /    /|
           o____. |
           |    | .          z
           |    |/       x__/
           .____.           |
                            y
           \endverbatim

           - there is one zero parameter: identifies one edge
           \verbatim
           e.g. direction<zero_, plus_, minus_> corresponds to:
             .____.
            /    /|
           .####. |
           |    | .
           |    |/
           .____.
           \endverbatim

           - there are 2 zero parameters: identifies one face
           \verbatim
           e.g. direction<zero_, zero_, minus_> corresponds to:
             .____.
            /    /|
           .____. |
           |####| .
           |####|/
           .####.
           \endverbatim
           - the case in which all three are zero does not belong to the boundary and is excluded.

           \tparam I_ Orientation in the I dimension
           \tparam J_ Orientation in the J dimension
           \tparam K_ Orientation in the K dimension
         */
        template <sign I_, sign J_, sign K_>
        struct direction {
            static constexpr sign i = I_;
            static constexpr sign j = J_;
            static constexpr sign k = K_;
        };
    } // namespace boundaries
} // namespace gridtools
