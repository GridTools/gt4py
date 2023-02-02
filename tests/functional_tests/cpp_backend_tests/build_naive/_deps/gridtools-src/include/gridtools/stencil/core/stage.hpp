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

#include "../../meta.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace stage_impl_ {
                template <class>
                struct meta_stage;

                template <template <class...> class L, class... Ts>
                struct meta_stage<L<Ts...>> : decltype(get_stage(std::declval<Ts>()...)) {};

                template <class Keys>
                struct make_stage_f {
                    template <class Functor>
                    using apply = typename meta_stage<typename Functor::param_list>::template apply<Functor, Keys>;
                };

            } // namespace stage_impl_
            template <class Functors, class Keys>
            using make_stages = meta::transform<stage_impl_::make_stage_f<Keys>::template apply, Functors>;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
