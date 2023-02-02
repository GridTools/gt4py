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

/** \file
    In this file the different types of architectures are
    defined. The Architectures specifies where the data to be
    exchanged by communication patterns are residing. The possible
    choices are: \link gridtools::gcl::cpu \endlink and \link gridtools::gcl::gpu.

    The assumption is that data to be exchanged is in the same place
    for all the processes involved in a pattern. That is, it is not
    possible to send data from a cpu main memory to a remote GPU
    memory.
*/

namespace gridtools {
    namespace gcl {
        /** Indicate that the data is on the main memory of the process
         */
        struct cpu {};

        /** Indicates that the data is on the memory of a GPU
         */
        struct gpu {};
    } // namespace gcl
} // namespace gridtools
