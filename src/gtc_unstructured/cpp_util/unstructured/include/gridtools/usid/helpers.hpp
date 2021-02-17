#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/utility.hpp>
#include <gridtools/meta/id.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/contiguous.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/sid/rename_dimensions.hpp>

#include "dim.hpp"
#include "domain.hpp"

namespace gridtools {
    namespace usid {
        template <class T, class Alloc, class HSize, class KSize>
        auto make_simple_tmp_storage(HSize h_size, KSize k_size, Alloc &alloc) {
            return sid::make_contiguous<T>(alloc, hymap::keys<dim::h, dim::k>::values<HSize, KSize>(h_size, k_size));
        }

        template <class T, class Alloc, class HSize, class KSize, class SparseSize>
        auto make_simple_sparse_tmp_storage(HSize h_size, KSize k_size, SparseSize s_size, Alloc &alloc) {
            return sid::make_contiguous<T>(
                alloc, hymap::keys<dim::h, dim::k, dim::s>::values<HSize, KSize, SparseSize>(h_size, k_size, s_size));
        }

        template <int_t N, bool HasSkipValues>
        struct connectivity {
            using max_neighbors_t = integral_constant<int_t, N>;
            using has_skip_values_t = std::integral_constant<bool, HasSkipValues>;
        };

        template <class Connectivity>
        struct sparse_field {
            using connectivity_t = Connectivity;
        };

        template <class, class = void>
        struct is_sparse_field : std::false_type {};

        template <class T>
        struct is_sparse_field<T, std::enable_if_t<std::is_base_of<sparse_field<typename T::connectivity_t>, T>::value>>
            : std::true_type {};

        template <class Connectivity, class Fun, class Ptr, class Strides, class Neighbors>
        GT_FUNCTION void foreach_neighbor(Fun &&fun, Ptr &&ptr, Strides &&strides, Neighbors &&neighbors) {
            sid::make_loop<Connectivity>(typename Connectivity::max_neighbors_t())([&](auto const &ptr, auto &&) {
                auto i_glob = *host_device::at_key<Connectivity>(ptr);
                if /*constexpr*/ (Connectivity::has_skip_values_t::value)
                    if (i_glob < 0)
                        return;
                fun(ptr, sid::shifted(neighbors.first, neighbors.second, i_glob));
            })(wstd::forward<decltype(ptr)>(ptr), strides);
        }

        template <class Connectivity, class Fun, class Ptr, class Strides, class Neighbors>
        GT_FUNCTION void foreach_neighbor_indexed(Fun &&fun, Ptr &&ptr, Strides &&strides, Neighbors &&neighbors) {
            int i_local = 0;
            sid::make_loop<Connectivity>(typename Connectivity::max_neighbors_t())([&](auto const &ptr, auto &&) {
                auto i_glob = *host_device::at_key<Connectivity>(ptr);
                if /*constexpr*/ (Connectivity::has_skip_values_t::value)
                    if (i_glob < 0)
                        return;
                fun(ptr, sid::shifted(neighbors.first, neighbors.second, i_glob), i_local);
                i_local++;
            })(wstd::forward<decltype(ptr)>(ptr), strides);
        }

        template <class Tag, class Val>
        decltype(auto) composite_item(Tag, Val &&val) {
            if /*constexpr*/ (is_sparse_field<Tag>::value)
                return sid::rename_dimensions<dim::s, typename Tag::connectivity_t>(std::forward<Val>(val));
            else
                return std::forward<Val>(val);
        }

        template <class... Tags>
        struct make_composite_f {
            template <class... Vals>
            auto operator()(Vals &&...vals) const {
                return sid::composite::make<Tags...>(composite_item(Tags(), std::forward<Vals>(vals))...);
            }
        };

        template <class... Tags>
        constexpr make_composite_f<Tags...> make_composite = {};
    } // namespace usid
} // namespace gridtools
