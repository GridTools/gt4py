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

#include <cassert>
#include <functional>
#include <utility>

#include "../../common/for_each.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../sid/sid_shift_origin.hpp"
#include "../be_api.hpp"
#include "convert_fe_to_be_spec.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace backend_impl_ {
                template <class Grid, class DataStores>
                auto shift_origin(Grid const &grid, DataStores data_stores) {
                    return tuple_util::transform(
                        [offsets = grid.origin()](
                            auto &&src) { return sid::shift_sid_origin(std::forward<decltype(src)>(src), offsets); },
                        std::move(data_stores));
                }

                template <class Spec>
                struct call_entry_point_f {
                    template <class Backend, class Grid, class DataStores>
                    void operator()(Backend &&be, Grid const &grid, DataStores data_stores) const {
                        using be_spec_t = convert_fe_to_be_spec<Spec, typename Grid::interval_t, DataStores>;
#ifndef NDEBUG
                        for_each<be_api::make_fused_view<be_spec_t>>([&](auto matrix) {
                            for_each<decltype(matrix)>([&](auto info) {
                                assert(((void)"domain k-size is too small", grid.k_size(info.interval()) >= 0));
                            });
                        });
#endif
                        gridtools_backend_entry_point(
                            std::forward<Backend>(be), be_spec_t(), grid, shift_origin(grid, std::move(data_stores)));
                    }
                };
            } // namespace backend_impl_
            using backend_impl_::call_entry_point_f;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
