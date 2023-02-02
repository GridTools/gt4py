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

#include <tuple>

#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "./column_stage.hpp"
#include "./run.hpp"
#include "./stencil_stage.hpp"

namespace gridtools::fn {
    namespace executor_impl_ {
        template <class Backend,
            int ArgOffset,
            class Sizes,
            class Offsets,
            class MakeIterator,
            class Args = std::tuple<>,
            class Specs = meta::list<>>
        struct executor_data {
            Backend m_backend;
            Sizes m_sizes;
            Offsets m_offsets;
            MakeIterator m_make_iterator;
            Args m_args = {};
            using arg_offset_t = std::integral_constant<int, ArgOffset>;
            using specs_t = Specs;

            template <class Arg>
            auto arg(Arg &&arg) && {
                auto args = tuple_util::deep_copy(
                    tuple_util::push_back(std::move(m_args), sid::shift_sid_origin(std::forward<Arg>(arg), m_offsets)));
                return executor_data<Backend, ArgOffset, Sizes, Offsets, MakeIterator, decltype(args), Specs>{
                    std::move(m_backend),
                    std::move(m_sizes),
                    std::move(m_offsets),
                    std::move(m_make_iterator),
                    std::move(args)};
            }

            template <class Spec>
            auto spec(Spec) && {
                using specs_t = meta::push_back<Specs, Spec>;
                return executor_data<Backend, ArgOffset, Sizes, Offsets, MakeIterator, Args, specs_t>{
                    std::move(m_backend),
                    std::move(m_sizes),
                    std::move(m_offsets),
                    std::move(m_make_iterator),
                    std::move(m_args)};
            }
        };

        template <class Data>
        struct stencil_executor {
            Data m_data;

            template <class Arg>
            auto arg(Arg &&arg) && {
                auto data = std::move(m_data).arg(std::forward<Arg>(arg));
                return stencil_executor<decltype(data)>{std::move(data)};
            }

            template <class Out, class Stencil, class... Ins>
            auto assign(Out, Stencil, Ins...) && {
                auto data = std::move(m_data).spec(stencil_stage<Stencil,
                    Out::value + Data::arg_offset_t::value,
                    Ins::value + Data::arg_offset_t::value...>());
                return stencil_executor<decltype(data)>{std::move(data)};
            }

            void execute() && {
                run_stencil_stages(std::move(m_data.m_backend),
                    typename Data::specs_t(),
                    std::move(m_data.m_make_iterator),
                    std::move(m_data.m_sizes),
                    std::move(m_data.m_args));
            }
        };

        template <class Vertical, class Data, class Seeds = std::tuple<>>
        struct vertical_executor {
            Data m_data;
            Seeds m_seeds = {};

            template <class Arg>
            auto arg(Arg &&arg) && {
                auto data = std::move(m_data).arg(std::forward<Arg>(arg));
                return vertical_executor<Vertical, decltype(data)>{std::move(data), std::move(m_seeds)};
            }

            template <class Out, class ScanOrFold, class Seed, class... Ins>
            auto assign(Out, ScanOrFold, Seed seed, Ins...) && {
                auto data = std::move(m_data).spec(column_stage<Vertical,
                    ScanOrFold,
                    Out::value + Data::arg_offset_t::value,
                    Ins::value + Data::arg_offset_t::value...>());
                auto seeds = tuple_util::deep_copy(tuple_util::push_back(std::move(m_seeds), std::move(seed)));
                return vertical_executor<Vertical, decltype(data), decltype(seeds)>{std::move(data), std::move(seeds)};
            }

            void execute() && {
                run_column_stages(std::move(m_data.m_backend),
                    typename Data::specs_t(),
                    std::move(m_data.m_make_iterator),
                    std::move(m_data.m_sizes),
                    Vertical(),
                    std::move(m_data.m_args),
                    std::move(m_seeds));
            }
        };

        // ArgOffset allows passing some args for backend usage while keeping them hidden from the user
        template <int ArgOffset = 0, class Backend, class Sizes, class Offsets, class MakeIterator>
        auto make_stencil_executor(
            Backend const &backend, Sizes const &sizes, Offsets const &offsets, MakeIterator const &make_iterator) {
            executor_data<Backend, ArgOffset, Sizes, Offsets, MakeIterator> data{
                backend, sizes, offsets, make_iterator};
            return stencil_executor<decltype(data)>{std::move(data)};
        }

        // ArgOffset allows passing some args for backend usage while keeping them hidden from the user
        template <class Vertical, int ArgOffset = 0, class Backend, class Sizes, class Offsets, class MakeIterator>
        auto make_vertical_executor(
            Backend const &backend, Sizes const &sizes, Offsets const &offsets, MakeIterator const &make_iterator) {
            executor_data<Backend, ArgOffset, Sizes, Offsets, MakeIterator> data{
                backend, sizes, offsets, make_iterator};
            return vertical_executor<Vertical, decltype(data)>{std::move(data)};
        }
    } // namespace executor_impl_

    using executor_impl_::make_stencil_executor;
    using executor_impl_::make_vertical_executor;
} // namespace gridtools::fn
