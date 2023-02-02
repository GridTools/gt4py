/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/hymap.hpp>

namespace gridtools {
    __device__ void test_compilation_issue_1715() {
        auto foo = [](auto &&i) -> auto && {
            hymap::keys<int>::make_values(i);
            return i;
        };
        foo(0);
    }
} // namespace gridtools
