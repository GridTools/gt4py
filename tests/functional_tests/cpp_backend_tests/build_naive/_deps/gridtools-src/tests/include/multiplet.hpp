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

#include <gridtools/common/array.hpp>
#include <gridtools/common/array_addons.hpp>

/**
   @brief Small value type to use in tests where we want to check the
   values in a fields, for instance to check if layouts works, on in
   communication tests
*/
template <gridtools::uint_t N>
using multiplet = gridtools::array<int, N>;

using triplet = multiplet<3>;
