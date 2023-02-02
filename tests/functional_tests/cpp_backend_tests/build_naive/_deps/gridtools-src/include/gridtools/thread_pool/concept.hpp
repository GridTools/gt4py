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

/*
 * The Concept of Thread Pool
 *
 * To be a thread pool a type should have the following functions to be available via ADL:
 *   thread_pool_get_thread_num(pool);
 *   thread_pool_get_max_threads(pool);
 *   thread_pool_parallel_for_loop(pool, func, limit);
 *
 *   where `pool` is an instance of the type.
 *
 *   There maybe additional overloads of `thread_pool_parallel_for_loop` avaliable:
 *   Note that the limits go from the inner most to the outer most
 *     thread_pool_parallel_for_loop(pool, func, lim0, lim1);
 *     thread_pool_parallel_for_loop(pool, func, lim0, lim1, lim2);
 *     etc.
 *   They are optional and could be provided for performance reasons.
 */

#include <tuple>

#include "../common/stride_util.hpp"
#include "../common/tuple_util.hpp"
namespace gridtools {
    namespace thread_pool {
        namespace concept_impl_ {
            template <class T>
            auto get_thread_num(T const &obj) -> decltype(thread_pool_get_thread_num(obj)) {
                return thread_pool_get_thread_num(obj);
            }

            template <class T>
            auto get_max_threads(T const &obj) -> decltype(thread_pool_get_max_threads(obj)) {
                return thread_pool_get_max_threads(obj);
            }

            template <class T, class F, class... Dims>
            auto thread_pool_parallel_for_loop(T const &obj, F const &f, Dims... limits)
                -> decltype(thread_pool_parallel_for_loop(obj, std::declval<void (*)(size_t)>, 0)) {
                std::tuple<Dims...> lims{limits...};
                thread_pool_parallel_for_loop(
                    obj,
                    [&, strides = stride_util::make_strides_from_sizes(lims)](auto index) {
                        tuple_util::apply(f,
                            tuple_util::transform(
                                [&](auto stride, auto lim) { return index / stride % lim; }, strides, lims));
                    },
                    stride_util::total_size(lims));
            }

            template <class T, class F, class... Dims>
            auto parallel_for_loop(T const &obj, F const &f, Dims... limits)
                -> decltype(thread_pool_parallel_for_loop(obj, f, limits...)) {
                return thread_pool_parallel_for_loop(obj, f, limits...);
            }
        } // namespace concept_impl_

        using concept_impl_::get_max_threads;
        using concept_impl_::get_thread_num;
        using concept_impl_::parallel_for_loop;
    } // namespace thread_pool
} // namespace gridtools
