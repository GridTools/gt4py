#pragma once

#include <gridtools/sid/concept.hpp>
#include <gridtools/storage/builder.hpp>

namespace gridtools {
    template <class Connectivity, class MaxNeighbors>
    auto make_connectivity_producer(MaxNeighbors max_neighbors, Connectivity src) {
        return [src](auto traits) {
            return gridtools::storage::builder<decltype(traits)>
                                .template type<int>()
                                .dimensions(at_key<integral_constant<int,0>>(sid::get_upper_bounds(src)), max_neighbors>{})
                                .initializer([&src](std::size_t p, std::size_t n){
                                    auto ptr = sid::get_origin(src)();
                                    auto strides = sid::get_strides(src);
                                    sid::shift(ptr, at_key<integral_constant<int, 0>>(strides), p);
                                    sid::shift(ptr, at_key<integral_constant<int, 1>>(strides), n);
                                    return *ptr - 1;})
                                .build();
        };
    };
} // namespace gridtools
