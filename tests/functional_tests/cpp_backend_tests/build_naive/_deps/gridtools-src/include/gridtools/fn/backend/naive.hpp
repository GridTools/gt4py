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

#include "../../common/functional.hpp"
#include "../../common/hymap.hpp"
#include "../../sid/allocator.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/unknown_kind.hpp"
#include "./common.hpp"

namespace gridtools::fn::backend {
    namespace naive_impl_ {
        struct naive {};

        template <class Sizes, class StencilStage, class MakeIterator, class Composite>
        void apply_stencil_stage(
            naive, Sizes const &sizes, StencilStage, MakeIterator &&make_iterator, Composite &&composite) {
            auto ptr = sid::get_origin(std::forward<Composite>(composite))();
            auto strides = sid::get_strides(std::forward<Composite>(composite));
            common::make_loops(sizes)([make_iterator = make_iterator()](auto ptr, auto const &strides) {
                StencilStage()(make_iterator, ptr, strides);
            })(ptr, strides);
        }

        template <class Sizes, class ColumnStage, class MakeIterator, class Composite, class Vertical, class Seed>
        void apply_column_stage(naive,
            Sizes const &sizes,
            ColumnStage,
            MakeIterator &&make_iterator,
            Composite &&composite,
            Vertical,
            Seed seed) {
            auto ptr = sid::get_origin(std::forward<Composite>(composite))();
            auto strides = sid::get_strides(std::forward<Composite>(composite));
            auto v_size = at_key<Vertical>(sizes);
            common::make_loops(hymap::canonicalize_and_remove_key<Vertical>(sizes))(
                [v_size = std::move(v_size), make_iterator = make_iterator(), seed = std::move(seed)](auto ptr,
                    auto const &strides) { ColumnStage()(seed, v_size, make_iterator, std::move(ptr), strides); })(
                ptr, strides);
        }

        inline auto tmp_allocator(naive be) { return std::tuple(be, sid::allocator(&std::make_unique<char[]>)); }

        template <class Allocator, class Sizes, class T>
        auto allocate_global_tmp(std::tuple<naive, Allocator> &alloc, Sizes const &sizes, data_type<T>) {
            return sid::make_contiguous<T, int_t, sid::unknown_kind>(std::get<1>(alloc), sizes);
        }
    } // namespace naive_impl_

    using naive_impl_::naive;

    using naive_impl_::apply_column_stage;
    using naive_impl_::apply_stencil_stage;

    using naive_impl_::allocate_global_tmp;
    using naive_impl_::tmp_allocator;
} // namespace gridtools::fn::backend
