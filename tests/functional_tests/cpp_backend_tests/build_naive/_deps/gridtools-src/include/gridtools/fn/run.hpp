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

#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/composite.hpp"
#include "./stencil_stage.hpp"

namespace gridtools::fn {
    namespace run_impl_ {
        template <class Sids>
        auto make_composite(Sids &&sids) {
            using keys_t = meta::iseq_to_list<std::make_integer_sequence<int, std::tuple_size_v<Sids>>,
                sid::composite::keys,
                integral_constant>;
            return tuple_util::convert_to<keys_t::template values>(std::forward<Sids>(sids));
        }

        template <class Backend, class StageSpecs, class MakeIterator, class Domain, class Sids>
        void run_stencil_stages(
            Backend const &backend, StageSpecs, MakeIterator const &make_iterator, Domain const &domain, Sids &&sids) {
            auto composite = make_composite(std::forward<Sids>(sids));
            tuple_util::for_each(
                [&](auto stage) { apply_stencil_stage(backend, domain, std::move(stage), make_iterator, composite); },
                meta::rename<std::tuple, StageSpecs>());
        }

        template <class Backend,
            class StageSpecs,
            class MakeIterator,
            class Domain,
            class Vertical,
            class Sids,
            class Seeds>
        void run_column_stages(Backend const &backend,
            StageSpecs,
            MakeIterator const &make_iterator,
            Domain const &domain,
            Vertical,
            Sids &&sids,
            Seeds &&seeds) {
            auto composite = make_composite(std::forward<Sids>(sids));
            tuple_util::for_each(
                [&](auto stage, auto seed) {
                    apply_column_stage(
                        backend, domain, std::move(stage), make_iterator, composite, Vertical(), std::move(seed));
                },
                meta::rename<std::tuple, StageSpecs>(),
                std::forward<Seeds>(seeds));
        }
    } // namespace run_impl_

    using run_impl_::run_column_stages;
    using run_impl_::run_stencil_stages;
} // namespace gridtools::fn
