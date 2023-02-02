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

#include "../common/hymap.hpp"
#include "../meta/logical.hpp"
#include "../sid/concept.hpp"
#include "../stencil/positional.hpp"
#include "./common_interface.hpp"
#include "./executor.hpp"
#include "./neighbor_table.hpp"

namespace gridtools::fn {
    namespace unstructured::dim {
        using horizontal = integral_constant<int, 0>;
        using vertical = integral_constant<int, 1>;
    } // namespace unstructured::dim

    namespace unstructured_impl_ {
        using gridtools::stencil::positional;
        namespace dim = unstructured::dim;

        template <class Tables, class Sizes>
        struct domain {
            Tables m_tables;
            Sizes m_sizes;

            Sizes const &sizes() const { return m_sizes; }
        };

        template <class Tables, class Sizes, class Offsets>
        struct domain_with_offsets : domain<Tables, Sizes> {
            Offsets m_offsets;

            static_assert(meta::all_of<neighbor_table::is_neighbor_table, tuple_util::traits::to_types<Tables>>());

            domain_with_offsets(Tables const &tables, Sizes const &sizes, Offsets const &offsets)
                : domain<Tables, Sizes>{tables, sizes}, m_offsets(offsets) {}

            domain<Tables, Sizes> const &without_offsets() const { return *this; }
        };

        template <class Tag, class NeighborTable>
        typename hymap::keys<Tag>::template values<NeighborTable> connectivity(NeighborTable const &nt) {
            static_assert(neighbor_table::is_neighbor_table<NeighborTable>());
            return {nt};
        }

        template <class Sizes = std::tuple<int, int>, class Offsets = std::tuple<>, class... Connectivities>
        auto unstructured_domain(Sizes const &sizes, Offsets const &offsets, Connectivities const &...conns) {

            return domain_with_offsets(hymap::concat(conns...), sizes, offsets);
        };

        template <class Tag, class Ptr, class Strides, class Domain>
        struct iterator {
            Ptr m_ptr;
            Strides const &m_strides;
            Domain const &m_domain;
            int m_index;
        };

        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION constexpr bool can_deref(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return it.m_index != -1;
        }

        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION constexpr auto deref(iterator<Tag, Ptr, Strides, Domain> const &it) {
            assert(can_deref(it));
            decltype(auto) stride = host_device::at_key<Tag>(sid::get_stride<dim::horizontal>(it.m_strides));
            return *sid::shifted(it.m_ptr, stride, it.m_index);
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Conn, class Offset>
        GT_FUNCTION constexpr auto horizontal_shift(iterator<Tag, Ptr, Strides, Domain> const &it, Conn, Offset) {
            auto const &table = host_device::at_key<Conn>(it.m_domain.m_tables);
            auto new_index = it.m_index == -1 ? -1 : get<Offset::value>(neighbor_table::neighbors(table, it.m_index));
            auto shifted = it;
            shifted.m_index = new_index;
            return shifted;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset>
        GT_FUNCTION constexpr auto non_horizontal_shift(
            iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset) {
            auto shifted = it;
            sid::shift(shifted.m_ptr, host_device::at_key<Tag>(sid::get_stride<Dim>(shifted.m_strides)), offset);
            return shifted;
        }

        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION constexpr auto shift(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return it;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset, class... Offsets>
        GT_FUNCTION constexpr auto shift(
            iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset, Offsets... offsets) {
            if constexpr (has_key<decltype(it.m_domain.m_tables), Dim>()) {
                return unstructured_impl_::shift(horizontal_shift(it, Dim(), offset), offsets...);
            } else {
                return unstructured_impl_::shift(non_horizontal_shift(it, Dim(), offset), offsets...);
            }
        }

        template <class Domain>
        struct make_iterator {
            Domain m_domain;

            explicit make_iterator(Domain const &domain) : m_domain(domain) {}

            GT_FUNCTION auto operator()() const {
                return [&](auto tag, auto const &ptr, auto const &strides) {
                    auto tptr = host_device::at_key<decltype(tag)>(ptr);
                    // the first argument is always the horizontal index
                    int index = *host_device::at_key<integral_constant<int, 0>>(ptr);
                    decltype(auto) stride =
                        host_device::at_key<decltype(tag)>(sid::get_stride<dim::horizontal>(strides));
                    sid::shift(tptr, stride, -index);
                    return iterator<decltype(tag), decltype(tptr), decltype(strides), Domain>{
                        std::move(tptr), strides, m_domain, index};
                };
            }
        };

        template <class Backend, class Domain, class TmpAllocator>
        struct backend {
            Backend m_backend;
            Domain m_domain;
            TmpAllocator m_allocator;

            static constexpr auto index = positional<dim::horizontal>();

            auto stencil_executor() const {
                return [&] {
                    return make_stencil_executor<1>(
                        m_backend, m_domain.m_sizes, m_domain.m_offsets, make_iterator(m_domain.without_offsets()))
                        .arg(index); // the horizontal index is passed as the first argument
                };
            }

            template <class Vertical = dim::vertical>
            auto vertical_executor(Vertical = {}) const {
                return [&] {
                    return make_vertical_executor<Vertical, 1>(
                        m_backend, m_domain.m_sizes, m_domain.m_offsets, make_iterator(m_domain.without_offsets()))
                        .arg(index); // the horizontal index is passed as the first argument
                };
            }
        };

        template <class Backend, class Tables, class Sizes, class Offsets>
        auto make_backend(Backend const &b, domain_with_offsets<Tables, Sizes, Offsets> const &d) {
            auto allocator = tmp_allocator(Backend());
            return backend<Backend, domain_with_offsets<Tables, Sizes, Offsets>, decltype(allocator)>{
                std::move(b), d, std::move(allocator)};
        }
    } // namespace unstructured_impl_

    using unstructured_impl_::can_deref;
    using unstructured_impl_::connectivity;
    using unstructured_impl_::deref;
    using unstructured_impl_::shift;
    using unstructured_impl_::unstructured_domain;
} // namespace gridtools::fn
