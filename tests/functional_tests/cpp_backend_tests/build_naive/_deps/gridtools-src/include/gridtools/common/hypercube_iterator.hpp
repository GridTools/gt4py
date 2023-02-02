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

#include <utility>

#include "array.hpp"
#include "defs.hpp"
#include "host_device.hpp"
#include "tuple_util.hpp"

namespace gridtools {
    namespace impl_ {

        template <size_t D>
        class hypercube_view {
          private:
            using point_t = array<size_t, D>;
            struct grid_iterator {
                point_t m_pos;
                const point_t &m_begin;
                const point_t &m_end;

                GT_FUNCTION grid_iterator &operator++() {
                    // The CUDA compiler is very sensitive in properly unrolling the following loop, please change with
                    // care. The kernel in test_layout_transformation_cuda should use 0 stack frame (and about 14
                    // registers) if this loop is properly unrolled.
                    bool continue_iterating = true;
                    for (size_t i = 0; i < D; ++i) {
                        if (continue_iterating) {
                            size_t index = D - i - 1;
                            if (m_pos[index] + 1 < m_end[index]) {
                                ++m_pos[index];
                                continue_iterating = false;
                            } else {
                                m_pos[index] = m_begin[index];
                            }
                        }
                    }
                    if (continue_iterating) // we reached the end of the iteration space
                        for (size_t i = 0; i < D; ++i)
                            m_pos[i] = m_end[i];
                    return *this;
                }

                GT_FUNCTION grid_iterator operator++(int) {
                    grid_iterator tmp(*this);
                    operator++();
                    return tmp;
                }

                GT_FUNCTION point_t const &operator*() const { return m_pos; }

                GT_FUNCTION bool operator!=(const grid_iterator &other) const { return m_pos != other.m_pos; }
            };

          public:
            GT_FUNCTION hypercube_view(const point_t &begin, const point_t &end) : m_begin(begin), m_end(end) {}
            GT_FUNCTION hypercube_view(const point_t &end) : m_end(end) {}

            GT_FUNCTION grid_iterator begin() const { return grid_iterator{m_begin, m_begin, m_end}; }
            GT_FUNCTION grid_iterator end() const { return grid_iterator{m_end, m_begin, m_end}; }

          private:
            point_t m_begin = {};
            point_t m_end;
        };

        template <class, class = void>
        struct is_pair_like : std::false_type {};

        template <class T>
        struct is_pair_like<T, std::enable_if_t<tuple_util::size<T>::value == 2>> : std::true_type {};

        template <class T>
        using is_size_t_like = std::is_convertible<size_t, T>;

    } // namespace impl_

    /**
     * @brief constructs a view on a hypercube from an array of ranges (e.g. pairs); the end of the range is exclusive.
     */
    template <typename Container,
        typename Decayed = std::decay_t<Container>,
        size_t OuterD = tuple_util::size<Decayed>::value,
        std::enable_if_t<OuterD != 0 && meta::all_of<impl_::is_pair_like, tuple_util::traits::to_types<Decayed>>::value,
            int> = 0>
    GT_FUNCTION impl_::hypercube_view<OuterD> make_hypercube_view(Container &&cube) {
        auto &&transposed = tuple_util::host_device::transpose(std::forward<Container>(cube));
        return {tuple_util::host_device::convert_to<array, size_t>(tuple_util::host_device::get<0>(transposed)),
            tuple_util::host_device::convert_to<array, size_t>(tuple_util::host_device::get<1>(transposed))};
    }

    /**
     * @brief short-circuit for zero dimensional hypercube (transpose cannot work)
     */
    template <typename Container,
        size_t D = tuple_util::size<std::decay_t<Container>>::value,
        std::enable_if_t<D == 0, int> = 0>
    GT_FUNCTION array<array<size_t, 0>, 1> make_hypercube_view(Container &&) {
        return {{}};
    }

    /**
     * @brief constructs a view on a hypercube from an array of integers (size of the loop in each dimension, ranges
     * start from 0); the end of the range is exclusive.
     */
    template <typename Container,
        typename Decayed = std::decay_t<Container>,
        size_t D = tuple_util::size<Decayed>::value,
        std::enable_if_t<D != 0 && meta::all_of<impl_::is_size_t_like, tuple_util::traits::to_types<Decayed>>::value,
            int> = 0>
    GT_FUNCTION impl_::hypercube_view<D> make_hypercube_view(Container &&sizes) {
        return {tuple_util::host_device::convert_to<array, size_t>(std::forward<Container>(sizes))};
    }
} // namespace gridtools
