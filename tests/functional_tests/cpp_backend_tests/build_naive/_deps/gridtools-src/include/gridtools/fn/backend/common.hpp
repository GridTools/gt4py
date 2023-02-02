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

#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/rename.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/loop.hpp"

namespace gridtools::fn::backend {

    namespace common {

        template <class Dims, class Sizes>
        constexpr GT_FUNCTION auto make_loops(Sizes const &sizes) {
            return tuple_util::host_device::fold(
                [&](auto outer, auto dim) {
                    return [outer = std::move(outer),
                               inner = sid::make_loop<decltype(dim)>(host_device::at_key<decltype(dim)>(sizes))](
                               auto &&...args) { return outer(inner(std::forward<decltype(args)>(args)...)); };
                },
                host_device::identity(),
                meta::rename<tuple, Dims>());
        }

        template <class Sizes>
        constexpr GT_FUNCTION auto make_loops(Sizes const &sizes) {
            return make_loops<get_keys<Sizes>>(sizes);
        }
    } // namespace common

    template <class T>
    struct data_type {};

} // namespace gridtools::fn::backend
