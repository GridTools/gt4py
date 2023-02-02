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

#include "../common/defs.hpp"
#include "../common/functional.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace loop_impl_ {

            template <class Key, class T>
            struct generic_loop {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                T m_num_steps;
                T m_step;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_num_steps;
                    T m_step;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &&ptr, Strides const &strides) const {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        auto &&stride = get_stride<Key>(strides);
                        for (T i = 0; i < m_num_steps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, m_step);
                        }
                        shift(std::forward<Ptr>(ptr), stride, -m_step * m_num_steps);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {std::forward<Fun>(fun), m_num_steps, m_step};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_num_steps;
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        if (++m_pos == m_num_steps) {
                            shift(ptr, get_stride<Key>(strides), m_step * (1 - m_num_steps));
                            m_pos = 0;
                            m_outer.next(std::forward<Ptr>(ptr), strides);
                        } else {
                            shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), m_step);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_num_steps <= 0 || m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {std::forward<Outer>(outer), m_num_steps, m_step, 0};
                }

                struct outer_most_cursor_f {
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        --m_pos;
                        shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), m_step);
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_step, m_num_steps}; }
            };

            template <class Key, class T, ptrdiff_t Step>
            struct known_step_loop {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                T m_num_steps;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_num_steps;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &&ptr, Strides const &strides) const {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        auto &&stride = get_stride<Key>(strides);
                        for (T i = 0; i < m_num_steps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, integral_constant<T, Step>{});
                        }
                        static constexpr T minus_step = -Step;
                        shift(std::forward<Ptr>(ptr), stride, minus_step * m_num_steps);
                    }
                };

                template <class Fun>
                GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {std::forward<Fun>(fun), m_num_steps};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_num_steps;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        if (++m_pos == m_num_steps) {
                            shift(ptr, get_stride<Key>(strides), Step * (1 - m_num_steps));
                            m_pos = 0;
                            m_outer.next(std::forward<Ptr>(ptr), strides);
                        } else {
                            shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), integral_constant<T, Step>{});
                        }
                    }

                    GT_FUNCTION bool done() const { return m_num_steps <= 0 || m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {std::forward<Outer>(outer), m_num_steps, 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        --m_pos;
                        shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), integral_constant<T, Step>{});
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_num_steps}; }
            };

            template <class Key, class T>
            struct known_step_loop<Key, T, 0> {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                T m_num_steps;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_num_steps;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &&ptr, const Strides &strides) const {
                        assert(m_num_steps >= 0);
                        for (T i = 0; i < m_num_steps; ++i)
                            m_fun(std::forward<Ptr>(ptr), strides);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {std::forward<Fun>(fun), m_num_steps};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_num_steps;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        if (++m_pos == m_num_steps) {
                            m_pos = 0;
                            m_outer.next(std::forward<Ptr>(ptr), strides);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_num_steps <= 0 || m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {std::forward<Outer>(outer), m_num_steps, 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&, Strides const &) {
                        --m_pos;
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_num_steps}; }
            };

            template <class Key, class T, T NumSteps>
            struct known_num_steps_loop {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                T m_step;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_step;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &&ptr, const Strides &strides) const {
                        auto &&stride = get_stride<Key>(strides);
                        // TODO(anstaf): to figure out if for_each<make_indices_c<NumSteps>>(...) produces better code.
                        for (T i = 0; i < NumSteps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, m_step);
                        }
                        static constexpr T minus_num_steps = -NumSteps;
                        shift(std::forward<Ptr>(ptr), stride, m_step * minus_num_steps);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {std::forward<Fun>(fun), m_step};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        if (++m_pos == NumSteps) {
                            constexpr T num_steps_back = 1 - NumSteps;
                            shift(ptr, get_stride<Key>(strides), m_step * num_steps_back);
                            m_pos = 0;
                            m_outer.next(std::forward<Ptr>(ptr), strides);
                        } else {
                            shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), m_step);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {std::forward<Outer>(outer), m_step, 0};
                }

                struct outer_most_cursor_f {
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        --m_pos;
                        shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), m_step);
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_step, NumSteps}; }
            };

            template <class Key, class T, ptrdiff_t NumSteps, ptrdiff_t Step>
            struct all_known_loop {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                template <class Fun>
                struct loop_f {
                    Fun m_fun;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &&ptr, const Strides &strides) const {
                        auto &&stride = get_stride<Key>(strides);
                        for (T i = 0; i < (T)NumSteps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, integral_constant<T, Step>{});
                        }
                        shift(std::forward<Ptr>(ptr), stride, integral_constant<T, -Step * NumSteps>{});
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {std::forward<Fun>(fun)};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        if (++m_pos == NumSteps) {
                            constexpr T offset_back = Step * (1 - NumSteps);
                            shift(ptr, get_stride<Key>(strides), offset_back);
                            m_pos = 0;
                            m_outer.next(std::forward<Ptr>(ptr), strides);
                        } else {
                            shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), Step);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {std::forward<Outer>(outer), 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        --m_pos;
                        shift(std::forward<Ptr>(ptr), get_stride<Key>(strides), Step);
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {NumSteps}; }
            };

            template <class Key, class T, ptrdiff_t NumSteps>
            struct all_known_loop<Key, T, NumSteps, 0> {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                template <class Fun>
                struct loop_f {
                    Fun m_fun;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &&ptr, Strides const &strides) const {
                        for (T i = 0; i < (T)NumSteps; ++i)
                            m_fun(ptr, strides);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {std::forward<Fun>(fun)};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&ptr, Strides const &strides) {
                        if (++m_pos == NumSteps) {
                            m_pos = 0;
                            m_outer.next(std::forward<Ptr>(ptr), strides);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {std::forward<Outer>(outer), 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&, Strides const &) {
                        --m_pos;
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {NumSteps}; }
            };

            template <class Key, class T>
            struct all_known_loop<Key, T, 1, 0> {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                template <class Fun>
                constexpr GT_FUNCTION Fun operator()(Fun &&fun) const {
                    return fun;
                }

                template <class Outer>
                constexpr GT_FUNCTION Outer make_cursor(Outer &&outer) const {
                    return std::forward<Outer>(outer);
                }

                struct outer_most_cursor_f {
                    bool m_done;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&, Strides const &) {
                        m_done = true;
                    }

                    GT_FUNCTION bool done() const { return m_done; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {false}; }
            };

            template <class Key, class T>
            struct all_known_loop<Key, T, 0, 0> {
                static_assert(std::is_signed_v<T>, GT_INTERNAL_ERROR);

                template <class Fun>
                constexpr GT_FUNCTION gridtools::host_device::noop operator()(Fun &&) const {
                    return {};
                }

                struct cursor_f {
                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &&, Strides const &) {}

                    GT_FUNCTION bool done() const { return true; }
                };

                template <class... Ts>
                constexpr GT_FUNCTION cursor_f make_cursor(Ts &&...) const {
                    return {};
                }
            };

            struct make_cursor_f {
                template <class Cursor, class Loop>
                constexpr GT_FUNCTION auto operator()(Cursor &&cursor, Loop const &loop) const {
                    return loop.make_cursor(std::forward<Cursor>(cursor));
                }
            };

            template <class Loops>
            constexpr GT_FUNCTION auto make_cursor_r(Loops &&loops) {
                return tuple_util::host_device::fold(make_cursor_f{},
                    tuple_util::host_device::get<0>(std::forward<Loops>(loops)).make_cursor(),
                    tuple_util::host_device::drop_front<1>(std::forward<Loops>(loops)));
            }

            template <class Loops>
            constexpr GT_FUNCTION auto make_cursor(Loops &&loops) {
                return make_cursor_r(tuple_util::host_device::reverse(std::forward<Loops>(loops)));
            }

            template <class Ptr, class Strides, class Cursor>
            struct range {
                Ptr m_ptr;
                Strides const &m_strides;
                Cursor m_cursor;

                GT_FUNCTION decltype(auto) operator*() const { return *m_ptr; }
                GT_FUNCTION void operator++() { m_cursor.next(m_ptr, m_strides); }
                template <class T>
                GT_FUNCTION bool operator!=(T &&) const {
                    return m_cursor.done();
                }

                range begin() const { return *this; }
                range end() const { return *this; }
            };

            template <class Ptr, class Strides, class Cursor>
            constexpr GT_FUNCTION range<Ptr, Strides const &, Cursor> make_range(
                Ptr ptr, Strides const &strides, Cursor &&cursor) {
                return {std::move(ptr), strides, std::forward<Cursor>(cursor)};
            }
        } // namespace loop_impl_

        /**
         *   A set of `make_loop<Key>(num_steps, step = 1)` overloads
         *
         *   @tparam I dimension index
         *   @param num_steps number of iterations in the loop. Can be of integral or integral_constant type
         *   @param step (optional) a step for each iteration. Can be of integral or integral_constant type.
         *               The default is integral_constant<int, 1>
         *   @return a functor that accepts another functor with the signature: `void(Ptr&&, Strides const&)` and
         *           returns a functor also with the same signature.
         *
         *   Usage:
         *     1. One dimensional traversal:
         *     ```
         *     // let us assume that we have a sid with stride dimension tags `i`, `j` and `k`.
         *
         *     // define the way we are going to traverse the data
         *     auto loop = sid::make_loop<k>(32);
         *
         *     // define what we are going to do with the data
         *     auto loop_body = [](auto& ptr, auto const& strides) { ... }
         *
         *     // bind traversal description with the body
         *     auto the_concrete_loop = loop(loop_body);
         *
         *     // execute the loop on the provided data
         *     the_concrete_loop(the_origin_of_my_data, the_strides_of_my_data);
         *     ```
         *
         *     2. Multi dimensional traversal:
         *     ```
         *     // define traversal path: k dimension is innermost and will be traversed backward
         *     auto multi_loop = compose(
         *       sid::make_loop<i>(i_size),
         *       sid::make_loop<j>(j_size),
         *       sid::make_loop<k>(k_size, -1_c));
         *
         *     // define what we are going to do with the data
         *     auto loop_body = [](auto& ptr, auto const& strides) { ... }
         *
         *     // bind traversal description with the body
         *     auto the_concrete_loop = multi_loop(loop_body);
         *
         *     auto ptr = the_origin_of_my_data;
         *     // first move the pointer to the end of data in k-direction
         *     sid::shift(ptr, sid::get_strides<2>(the_strides_of_my_data), 1_c - k_size);
         *
         *     // execute the loop on the provided data
         *     the_concrete_loop(ptr, the_strides_of_my_data);
         *     ```
         *   Rationale:
         *
         *     The goal of the design is to separate traversal description (dimensions order, numbers of steps,
         *     traversal directions), the body of the loop and the structure of the concrete data (begin point, strides)
         *     into orthogonal components.
         *
         *   Overloads:
         *
         *      `make_loop` goes with the large number of overloads to benefit from the fact that some aspects of
         *      traversal description are known in compile time.
         */
        template <class Key,
            class T1,
            class T2,
            class T = std::common_type_t<T1, T2>,
            std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>, int> = 0>
        constexpr GT_FUNCTION loop_impl_::generic_loop<Key, std::make_signed_t<T>> make_loop(T1 num_steps, T2 step) {
            return {std::make_signed_t<T>(num_steps), step};
        }

        template <class Key,
            class T1,
            class T2 = int,
            T2 Step = 1,
            class T = std::common_type_t<T1, T2>,
            std::enable_if_t<std::is_integral_v<T1>, int> = 0>
        constexpr GT_FUNCTION loop_impl_::known_step_loop<Key, std::make_signed_t<T>, Step> make_loop(
            T1 num_steps, std::integral_constant<T2, Step> = {}) {
            return {std::make_signed_t<T>(num_steps)};
        }

        template <class Key,
            class T1,
            T1 NumStepsV,
            class T2,
            class T = std::common_type_t<T1, T2>,
            std::enable_if_t<std::is_integral_v<T1> && (NumStepsV > 1), int> = 0>
        constexpr GT_FUNCTION loop_impl_::known_num_steps_loop<Key, std::make_signed_t<T>, NumStepsV> make_loop(
            std::integral_constant<T1, NumStepsV>, T2 step) {
            return {step};
        }

        template <class Key,
            class T1,
            T1 NumStepsV,
            class T2,
            class T = std::common_type_t<T1, T2>,
            std::enable_if_t<std::is_integral_v<T1> && (NumStepsV == 0 || NumStepsV == 1), int> = 0>
        constexpr GT_FUNCTION loop_impl_::all_known_loop<Key, std::make_signed_t<T>, NumStepsV, 0> make_loop(
            std::integral_constant<T1, NumStepsV>, T2) {
            return {};
        }

        template <class Key,
            class T1,
            T1 NumStepsV,
            class T2 = int,
            T2 StepV = 1,
            class T = std::common_type_t<T1, T2>,
            std::enable_if_t<(NumStepsV >= 0), int> = 0>
        constexpr GT_FUNCTION
            loop_impl_::all_known_loop<Key, std::make_signed_t<T>, NumStepsV, (NumStepsV > 1) ? StepV : 0>
            make_loop(std::integral_constant<T1, NumStepsV>, std::integral_constant<T2, StepV> = {}) {
            return {};
        }

        /**
         *   A helper that allows to use `SID`s with C++11 range based loop
         *
         *   Example:
         *
         *   using namespace gridtools::sid;
         *
         *   double data[3][4][5];
         *
         *   for(auto& ref : make_range(get_origin(data), get_strides(data),
         *                              make_loop<i>(3_c), make_loop<j>(4_c), make_loop<k>(5_c))) {
         *     ref = 42;
         *   }
         */
        template <class Ptr, class Strides, class OuterMostLoop, class... Loops>
        constexpr GT_FUNCTION auto make_range(
            Ptr ptr, Strides const &strides, OuterMostLoop &&outer_most_loop, Loops &&...loops) {
            return loop_impl_::make_range(std::move(ptr),
                strides,
                loop_impl_::make_cursor(tuple<OuterMostLoop, Loops...>{
                    std::forward<OuterMostLoop>(outer_most_loop), std::forward<Loops>(loops)...}));
        }

    } // namespace sid
} // namespace gridtools
