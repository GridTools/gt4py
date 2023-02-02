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

namespace gridtools {
    template <class F>
    constexpr F &&compose(F &&f) {
        return std::forward<F>(f);
    }

    template <class F, class... Fs>
    constexpr auto compose(F &&f, Fs &&...fs) {
        return [f = std::forward<F>(f), fs = compose(std::forward<Fs>(fs)...)](
                   auto &&...args) { return f(fs(std::forward<decltype(args)>(args)...)); };
    }
} // namespace gridtools
