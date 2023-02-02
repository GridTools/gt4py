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

#include <cassert>
#include <type_traits>
#include <utility>

#include "../common/functional.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta/is_instantiation_of.hpp"
#include "../sid/concept.hpp"

namespace gridtools::fn {
    namespace column_stage_impl_ {
        template <class F, class Projector = host_device::identity>
        struct scan_pass {
            F m_f;
            Projector m_p;
            constexpr GT_FUNCTION scan_pass(F f, Projector p = {}) : m_f(f), m_p(p) {}
        };

        template <class T>
        using is_scan_pass = meta::is_instantiation_of<scan_pass, T>;

        template <bool IsBackward>
        struct base : std::bool_constant<IsBackward> {
            static GT_FUNCTION constexpr auto prologue() { return tuple<>(); }
            static GT_FUNCTION constexpr auto epilogue() { return tuple<>(); }
        };

        using fwd = base<false>;
        using bwd = base<true>;

        template <class Vertical, class ScanOrFold, int Out, int... Ins>
        struct column_stage {
            template <class Seed, class MakeIterator, class Ptr, class Strides>
            GT_FUNCTION auto operator()(
                Seed seed, std::size_t size, MakeIterator &&make_iterator, Ptr ptr, Strides const &strides) const {
                constexpr std::size_t prologue_size = std::tuple_size_v<decltype(ScanOrFold::prologue())>;
                constexpr std::size_t epilogue_size = std::tuple_size_v<decltype(ScanOrFold::epilogue())>;
                assert(size >= prologue_size + epilogue_size);
                using step_t = integral_constant<int, ScanOrFold::value ? -1 : 1>;
                auto const &v_stride = sid::get_stride<Vertical>(strides);
                auto inc = [&] { sid::shift(ptr, v_stride, step_t()); };
                auto next = [&](auto acc, auto pass) {
                    if constexpr (is_scan_pass<decltype(pass)>()) {
                        // scan
                        auto res =
                            pass.m_f(std::move(acc), make_iterator(integral_constant<int, Ins>(), ptr, strides)...);
                        *host_device::at_key<integral_constant<int, Out>>(ptr) = pass.m_p(res);
                        inc();
                        return res;
                    } else {
                        // fold
                        auto res = pass(std::move(acc), make_iterator(integral_constant<int, Ins>(), ptr, strides)...);
                        inc();
                        return res;
                    }
                    // disable incorrect warning "missing return statement at end of non-void function"
                    GT_NVCC_DIAG_PUSH_SUPPRESS(940)
                };
                GT_NVCC_DIAG_POP_SUPPRESS(940)
                if constexpr (ScanOrFold::value)
                    sid::shift(ptr, v_stride, size - 1);
                auto acc = tuple_util::host_device::fold(next, std::move(seed), ScanOrFold::prologue());
                std::size_t n = size - prologue_size - epilogue_size;
                for (std::size_t i = 0; i < n; ++i)
                    acc = next(std::move(acc), ScanOrFold::body());
                return tuple_util::host_device::fold(next, std::move(acc), ScanOrFold::epilogue());
            }
        };

        template <class... ColumnStages>
        struct merged_column_stage {
            template <class Seed, class MakeIterator, class Ptr, class Strides>
            GT_FUNCTION auto operator()(
                Seed seed, std::size_t size, MakeIterator &&make_iterator, Ptr ptr, Strides const &strides) const {
                return tuple_util::host_device::fold(
                    [&](auto acc, auto stage) {
                        return stage(std::move(acc), size, std::forward<MakeIterator>(make_iterator), ptr, strides);
                    },
                    std::move(seed),
                    tuple(ColumnStages()...));
            }
        };
    } // namespace column_stage_impl_

    using column_stage_impl_::bwd;
    using column_stage_impl_::column_stage;
    using column_stage_impl_::fwd;
    using column_stage_impl_::merged_column_stage;
    using column_stage_impl_::scan_pass;
} // namespace gridtools::fn
