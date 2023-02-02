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
#include <type_traits>
#include <utility>
#include <vector>

#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/for_each.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil/frontend/icosahedral/connectivity.hpp>
#include <gridtools/stencil/frontend/icosahedral/location_type.hpp>

namespace gridtools {
    namespace _impl {
        struct neighbour {
            int_t i, j, k, c;
            template <class Fun>
            auto call(Fun &&fun) const {
                return std::forward<Fun>(fun)(i, j, k, c);
            }
        };

        template <class FromLocation, class ToLocation>
        std::vector<array<int_t, 4>> get_offsets(int_t, std::integral_constant<int_t, FromLocation::value>) {
            assert(false);
            return {};
        }

        template <class FromLocation,
            class ToLocation,
            int_t C = 0,
            std::enable_if_t<(C < FromLocation::value), int> = 0>
        std::vector<array<int_t, 4>> get_offsets(int_t c, std::integral_constant<int_t, C> = {}) {
            if (c > C) {
                return get_offsets<FromLocation, ToLocation>(c, std::integral_constant<int_t, C + 1>{});
            }
            std::vector<array<int_t, 4>> res;
            for_each<stencil::icosahedral::neighbor_offsets<FromLocation, ToLocation, C>>([&](auto src) {
                using tuple_util::get;
                res.push_back({get<0>(src), get<1>(src), get<2>(src), get<3>(src) - C});
            });
            return res;
        }
    } // namespace _impl

    template <class FromLocation, class ToLocation>
    std::vector<_impl::neighbour> neighbours_of(int_t i, int_t j, int_t k, int_t c) {
        assert(c >= 0);
        assert(c < FromLocation::value);
        std::vector<_impl::neighbour> res;
        for (auto &item : _impl::get_offsets<FromLocation, ToLocation>(c))
            res.push_back({i + item[0], j + item[1], k + item[2], c + item[3]});
        return res;
    }
} // namespace gridtools
