#pragma once

#include <atlas/mesh/Connectivity.h>

#include <gridtools/storage/builder.hpp>

namespace atlas namespace {
    mesh {
        template <class Connectivity, class MaxNeighbors>
        auto make_storage_producer(MaxNeighbors max_neighbors, Connectivity const &src) {
            return [&src, max_neighbors](auto traits) {
                return gridtools::storage::builder<decltype(traits)>
            .template type<int>()
            .dimensions(src.rows(), max_neighbors)
            .initializer([&src](auto row, auto col) { return col < src.cols(row) ? src(row, col) : -1; })
            .build();
            };
        }
    }
} // namespace namespace
