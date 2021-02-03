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

namespace gridtools::usid {
    template <class T, class Alloc, class HSize, class KSize>
    auto make_simple_tmp_storage(HSize h_size, KSize k_size, Alloc &alloc) {
        return sid::make_contiguous<T>(alloc, hymap::keys<dim::h, dim::k>::values<HSize, KSize>(h_size, k_size));
    }

    template <int_t N, bool HasSkipValues>
    struct connectivity {
        using max_neighbors_t = integral_constant<int_t, N>;
        using has_skip_values_t = std::bool_constant<HasSkipValues>;
    };

    template <class Connectivity>
    struct sparse_field {
        using connectivity_t = Connectivity;
    };

    template <class, class = void>
    struct is_sparse_field : std::false_type {};

    template <class T>
    struct is_sparse_field<T, std::enable_if_t<std::is_base_of_v<sparse_field<typename T::connectivity_t>, T>>>
        : std::true_type {};

    template <class T, class Conncectivity, class F, class Init, class G, class Ptr, class Strides, class Neighbors>
    GT_FUNCTION T fold_neighbors(F f, Init init, G g, Ptr &&ptr, Strides &&strides, Neighbors &&neighbors) {
        T acc = init(meta::lazy::id<T>());
        sid::make_loop<Conncectivity>(typename Conncectivity::max_neighbors_t())([&](auto const &ptr, auto &&) {
            auto i = *host_device::at_key<Conncectivity>(ptr);
            if constexpr (Conncectivity::has_skip_values_t::value)
                if (i < 0)
                    return;
            acc = f(acc, g(ptr, sid::shifted(neighbors.first, neighbors.second, i)));
        })(wstd::forward<decltype(ptr)>(ptr), strides);
        return acc;
    }

    template <class T, class Connectivity, class F, class... Args>
    GT_FUNCTION T sum_neighbors(F f, Args &&...args) {
        return fold_neighbors<T, Connectivity>([](auto x, auto y) { return x + y; },
            [](auto z) -> typename decltype(z)::type { return 0; },
            f,
            wstd::forward<Args>(args)...);
    }

    template <class T, class Connectivity, class F, class... Args>
    GT_FUNCTION T product_neighbors(F f, Args &&...args) {
        return fold_neighbors<T, Connectivity>([](auto x, auto y) { return x * y; },
            [](auto z) -> typename decltype(z)::type { return 1; },
            f,
            wstd::forward<Args>(args)...);
    }

    template <class T, class Connectivity, class F, class... Args>
    GT_FUNCTION T min_neighbors(F f, Args &&...args) {
        return fold_neighbors<T, Connectivity>([](auto x, auto y) { return x < y ? x : y; },
            [](auto z) -> typename decltype(z)::type {
                constexpr auto res = std::numeric_limits<typename decltype(z)::type>::max();
                return res;
            },
            f,
            wstd::forward<Args>(args)...);
    }

    template <class T, class Connectivity, class F, class... Args>
    GT_FUNCTION T max_neighbors(F f, Args &&...args) {
        return fold_neighbors<T, Connectivity>([](auto x, auto y) { return x > y ? x : y; },
            [](auto z) -> typename decltype(z)::type {
                constexpr auto res = std::numeric_limits<typename decltype(z)::type>::min();
                return res;
            },
            f,
            wstd::forward<Args>(args)...);
    }

    template <class Tag, class Val>
    decltype(auto) composite_item(Tag, Val &&val) {
        if constexpr (is_sparse_field<Tag>::value)
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
} // namespace gridtools::usid
