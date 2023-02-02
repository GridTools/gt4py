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

#include "concat.hpp"
#include "curry.hpp"
#include "fold.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "push_back.hpp"
#include "rename.hpp"
#include "transform.hpp"

namespace gridtools {
    namespace meta {
        template <class L>
        struct cartesian_product_step_impl_impl {
            template <class T>
            using apply = transform<curry<push_back, T>::template apply, rename<list, L>>;
        };

        template <class S, class L>
        using cartesian_product_step_impl =
            rename<concat, transform<cartesian_product_step_impl_impl<L>::template apply, S>>;

        template <class... Lists>
        using cartesian_product = foldl<cartesian_product_step_impl, list<list<>>, list<Lists...>>;
    } // namespace meta
} // namespace gridtools
