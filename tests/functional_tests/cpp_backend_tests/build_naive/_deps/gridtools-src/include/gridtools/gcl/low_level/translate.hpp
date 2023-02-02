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

#include <utility>

#include "../../common/layout_map.hpp"

namespace gridtools {
    namespace gcl {
        template <int D, class Is = std::make_integer_sequence<int, D>>
        struct default_layout_map;

        template <int D, int... Is>
        struct default_layout_map<D, std::integer_sequence<int, Is...>> {
            using type = layout_map<Is...>;
        };

        template <int D, typename map = typename default_layout_map<D>::type>
        struct translate_t;

        template <>
        struct translate_t<2, layout_map<0, 1>> {
            typedef layout_map<0, 1> map_type;
            inline int operator()(int I, int J) { return (I + 1) * 3 + J + 1; }
        };

        template <>
        struct translate_t<2, layout_map<1, 0>> {
            typedef layout_map<1, 0> map_type;
            inline int operator()(int I, int J) { return (J + 1) * 3 + I + 1; }
        };

        template <>
        struct translate_t<3, layout_map<0, 1, 2>> {
            typedef layout_map<0, 1, 2> map_type;
            inline int operator()(int I, int J, int K) { return (K + 1) * 9 + (J + 1) * 3 + I + 1; }
        };

        template <>
        struct translate_t<3, layout_map<2, 1, 0>> {
            typedef layout_map<2, 1, 0> map_type;
            inline int operator()(int I, int J, int K) { return (I + 1) * 9 + (J + 1) * 3 + K + 1; }
        };

        template <>
        struct translate_t<3, layout_map<1, 2, 0>> {
            typedef layout_map<1, 2, 0> map_type;
            inline int operator()(int I, int J, int K) { return (J + 1) * 9 + (I + 1) * 3 + K + 1; }
        };

        template <>
        struct translate_t<3, layout_map<0, 2, 1>> {
            typedef layout_map<0, 2, 1> map_type;
            inline int operator()(int I, int J, int K) { return (K + 1) * 9 + (I + 1) * 3 + J + 1; }
        };

        template <>
        struct translate_t<3, layout_map<2, 0, 1>> {
            typedef layout_map<2, 0, 1> map_type;
            inline int operator()(int I, int J, int K) { return (I + 1) * 9 + (K + 1) * 3 + J + 1; }
        };

        template <>
        struct translate_t<3, layout_map<1, 0, 2>> {
            typedef layout_map<1, 0, 2> map_type;
            inline int operator()(int I, int J, int K) { return (J + 1) * 9 + (K + 1) * 3 + I + 1; }
        };
    } // namespace gcl
} // namespace gridtools
