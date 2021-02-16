#pragma once

#include <memory>
#include <tuple>
#include <utility>

#include <gridtools/common/compose.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>

#include "dim.hpp"
#include "helpers.hpp"

namespace gridtools {
namespace usid {
namespace naive {
    using traits_t = storage::cpu_ifirst;

    inline auto make_allocator() { return sid::make_cached_allocator(&std::make_unique<char[]>); }

    template <class Kernel, class HSize, class KSize, class Sid, class... Sids>
    void call_kernel(HSize h_size, KSize k_size, Sid &&fields, Sids &&...neighbor_fields) {
        compose(sid::make_loop<dim::h>(h_size), sid::make_loop<dim::k>(k_size))([&](auto &ptr, auto const &strides) {
            Kernel()()(ptr,
                strides,
                std::make_pair(
                    sid::get_origin(neighbor_fields)(), sid::get_stride<dim::h>(sid::get_strides(neighbor_fields)))...);
        })(sid::get_origin(fields)(), sid::get_strides(fields));
    }

    template <class Tag, class Ptr>
    decltype(auto) field(Ptr const &ptr) {
        return *at_key<Tag>(ptr);
    }
}
}
}
