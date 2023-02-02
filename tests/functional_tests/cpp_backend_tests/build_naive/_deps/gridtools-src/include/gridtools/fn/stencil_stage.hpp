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

#include "../common/hymap.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple_util.hpp"

namespace gridtools::fn {

    template <class Stencil, int Out, int... Ins>
    struct stencil_stage {
        template <class MakeIterator, class Ptr, class Strides>
        GT_FUNCTION void operator()(MakeIterator &&make_iterator, Ptr &ptr, Strides const &strides) const {
            *host_device::at_key<integral_constant<int, Out>>(ptr) =
                Stencil()()(make_iterator(integral_constant<int, Ins>(), ptr, strides)...);
        }
    };

    template <class... Stages>
    struct merged_stencil_stage {
        template <class MakeIterator, class Ptr, class Strides>
        GT_FUNCTION void operator()(MakeIterator &&make_iterator, Ptr &ptr, Strides const &strides) const {
            (Stages()(std::forward<MakeIterator>(make_iterator), ptr, strides), ...);
        }
    };

} // namespace gridtools::fn
