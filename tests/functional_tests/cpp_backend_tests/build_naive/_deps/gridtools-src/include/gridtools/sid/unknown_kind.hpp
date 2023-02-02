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

namespace gridtools {
    namespace sid {
        // If for some SID `sid_get_strides_kind(SID const&)` returns `unknown_kind`,
        // it has the special meaning: SID implementation does not guarantee that the strides are the same
        // for any instances of that SID.
        //
        // The practical advice is the following:
        // If your application uses `sid::composite` and your fields model SIDs you have a choice:
        //  - meaningfully choose strides kinds for your fields in order benefit from sid::composite strides
        //    compression logic. In this case it is on your responsibility to ensure that the strides are the same
        //    if the kinds are the same.
        //  - use `unknown_kind` if you want to opt out your field from sid::composite optimizations.
        struct unknown_kind {};
    } // namespace sid
} // namespace gridtools
