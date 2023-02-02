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

#include <functional>

#include "../common/tuple_util.hpp"
#include "../sid/concept.hpp"
#include "./common_interface.hpp"
#include "./executor.hpp"

namespace gridtools::fn {
    namespace cartesian::dim {
        using i = integral_constant<int, 0>;
        using j = integral_constant<int, 1>;
        using k = integral_constant<int, 2>;
    } // namespace cartesian::dim

    namespace cartesian_impl_ {
        namespace dim = cartesian::dim;

        template <class Sizes, class Offsets = std::tuple<>>
        struct cartesian_domain {
            Sizes m_sizes;
            Offsets m_offsets;

            cartesian_domain(Sizes const &sizes, Offsets const &offsets = {}) : m_sizes(sizes), m_offsets(offsets) {}

            Sizes const &sizes() const { return m_sizes; }
        };

        template <class Tag, class Ptr, class Strides>
        struct iterator {
            Ptr m_ptr;
            Strides const &m_strides;
        };

        template <class Tag, class Ptr, class Strides>
        GT_FUNCTION auto deref(iterator<Tag, Ptr, Strides> const &it) {
            return *it.m_ptr;
        }

        template <class Tag, class Ptr, class Strides, class Dim, class Offset, class... Offsets>
        GT_FUNCTION void shift_impl(iterator<Tag, Ptr, Strides> &it, Dim, Offset offset, Offsets... offsets) {
            sid::shift(it.m_ptr, host_device::at_key<Tag>(sid::get_stride<Dim>(it.m_strides)), offset);
            shift_impl(it, offsets...);
        }
        template <class Tag, class Ptr, class Strides>
        GT_FUNCTION void shift_impl(iterator<Tag, Ptr, Strides> &) {}

        template <class Tag, class Ptr, class Strides, class... Offsets>
        GT_FUNCTION auto shift(iterator<Tag, Ptr, Strides> it, Offsets... offsets) {
            shift_impl(it, offsets...);
            return it;
        }

        struct make_iterator {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &strides) {
                    auto tptr = host_device::at_key<decltype(tag)>(ptr);
                    return iterator<decltype(tag), decltype(tptr), decltype(strides)>{std::move(tptr), strides};
                };
            }
        };

        template <class Backend, class Domain, class TmpAllocator>
        struct backend {
            Backend m_backend;
            Domain m_domain;
            TmpAllocator m_allocator;

            auto stencil_executor() const {
                return [&] {
                    return make_stencil_executor(m_backend, m_domain.m_sizes, m_domain.m_offsets, make_iterator());
                };
            }

            template <class Vertical = dim::k>
            auto vertical_executor(Vertical = {}) const {
                return [&] {
                    return make_vertical_executor<Vertical>(
                        m_backend, m_domain.m_sizes, m_domain.m_offsets, make_iterator());
                };
            }
        };

        template <class Backend, class Sizes, class Offsets>
        auto make_backend(Backend const &b, cartesian_domain<Sizes, Offsets> const &d) {
            auto allocator = tmp_allocator(Backend());
            return backend<Backend, cartesian_domain<Sizes, Offsets>, decltype(allocator)>{b, d, std::move(allocator)};
        }
    } // namespace cartesian_impl_
    using cartesian_impl_::cartesian_domain;
    using cartesian_impl_::deref;
    using cartesian_impl_::make_backend;
    using cartesian_impl_::shift;
} // namespace gridtools::fn
