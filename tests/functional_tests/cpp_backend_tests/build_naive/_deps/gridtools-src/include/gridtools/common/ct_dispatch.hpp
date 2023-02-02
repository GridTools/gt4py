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

/***
 *  `ct_dispatch` performs compile time dispatch on runtime value
 *
 *  Usage:
 *
 *  Say you have a function that accepts an integer in compile time:
 *
 *  template <size_t N> int foo() { ... }
 *
 *  And you need something like:
 *
 *  auto bar(int n) {
 *      switch(n) {
 *      case 0:
 *          return foo<0>();
 *      case 1:
 *          return foo<1>();
 *      case 2:
 *          return foo<2>();
 *      case 3:
 *          return foo<3>();
 *      }
 *  }
 *
 *  You can use `ct_dispatch` here to reduce the boilerplate:
 *
 *  auto bar(int n) {
 *      return ct_dispatch<4>([](auto n) {
 *          return foo<decltype(n)::value>();)
 *      }, n);
 *  }
 *
 */

#include <cassert>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace gridtools {
    template <size_t Lim, class Sink, std::enable_if_t<Lim == 1, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n == 0);
        return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
    }

    template <size_t Lim, class Sink, std::enable_if_t<Lim == 2, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < 2);
        switch (n) {
        case 0:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
        default:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 1>());
        }
    }

    template <size_t Lim, class Sink, std::enable_if_t<Lim == 3, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < 3);
        switch (n) {
        case 0:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
        case 1:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 1>());
        default:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 2>());
        }
    }

    template <size_t Lim, class Sink, std::enable_if_t<Lim == 4, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < 4);
        switch (n) {
        case 0:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
        case 1:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 1>());
        case 2:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 2>());
        default:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 3>());
        }
    }

    template <size_t Lim, class Sink, std::enable_if_t<Lim == 5, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < 5);
        switch (n) {
        case 0:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
        case 1:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 1>());
        case 2:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 2>());
        case 3:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 3>());
        default:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 4>());
        }
    }

    template <size_t Lim, class Sink, std::enable_if_t<Lim == 6, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < 6);
        switch (n) {
        case 0:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
        case 1:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 1>());
        case 2:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 2>());
        case 3:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 3>());
        case 4:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 4>());
        default:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 5>());
        }
    }

    template <size_t Lim, class Sink, std::enable_if_t<Lim == 7, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < 7);
        switch (n) {
        case 0:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
        case 1:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 1>());
        case 2:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 2>());
        case 3:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 3>());
        case 4:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 4>());
        case 5:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 5>());
        default:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 6>());
        }
    }

    template <size_t Lim, class Sink, std::enable_if_t<Lim == 8, int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < 8);
        switch (n) {
        case 0:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 0>());
        case 1:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 1>());
        case 2:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 2>());
        case 3:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 3>());
        case 4:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 4>());
        case 5:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 5>());
        case 6:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 6>());
        default:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, 7>());
        }
    }

    template <size_t Lim, class Sink, std::enable_if_t<(Lim > 8), int> = 0>
    auto ct_dispatch(Sink &&sink, size_t n) {
        assert(n < Lim);
        switch (n) {
        case Lim - 8:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 8>());
        case Lim - 7:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 7>());
        case Lim - 6:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 6>());
        case Lim - 5:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 5>());
        case Lim - 4:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 4>());
        case Lim - 3:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 3>());
        case Lim - 2:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 2>());
        case Lim - 1:
            return std::forward<Sink>(sink)(std::integral_constant<size_t, Lim - 1>());
        default:
            return ct_dispatch<Lim - 8>(std::forward<Sink>(sink), n);
        }
    }
} // namespace gridtools
