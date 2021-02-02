#pragma once

#include <cstddef>

#include "gridtools/common/layout_map.hpp"
#include <atlas/mesh.h>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include <mesh/Connectivity.h>

#include "mesh.hpp"
#include "unstructured.hpp"
#include <gridtools/next/atlas_field_util.hpp>

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace gridtools::next::atlas_wrappers {
    // not really a connectivity
    template <class LocationType>
    struct primary_connectivity {
        std::size_t size_;

        GT_FUNCTION friend std::size_t connectivity_size(primary_connectivity const &conn) { return conn.size_; }
    };

    template <class LocationType, std::size_t MaxNeighbors>
    struct regular_connectivity {
        struct builder {
            auto operator()(std::size_t size) {
                return gridtools::storage::builder<storage_trait>.template type<int>().template layout<0, 1>().template dimensions(
                    size, std::integral_constant<std::size_t, MaxNeighbors>{});
            }
        };

        decltype(builder{}(std::size_t{})()) tbl_;
        const atlas::idx_t missing_value_; // TODO Not sure if we can leave the type open
        const gridtools::uint_t size_;

        regular_connectivity(atlas::mesh::IrregularConnectivity const &conn)
            : tbl_{builder{}(conn.rows()).initializer([&conn](std::size_t row, std::size_t col) {
                  return col < static_cast<size_t>(conn.cols(row)) ? conn.row(row)(col) : conn.missing_value();
              })()},
              missing_value_{conn.missing_value()}, size_{tbl_->lengths()[0]} {}

        regular_connectivity(atlas::mesh::MultiBlockConnectivity const &conn)
            : tbl_{builder{}(conn.rows()).initializer([&conn](std::size_t row, std::size_t col) {
                  return col < static_cast<std::size_t>(conn.cols(row)) ? conn.row(row)(col) : conn.missing_value();
              })()},
              missing_value_{conn.missing_value()}, size_{tbl_->lengths()[0]} {}

        GT_FUNCTION friend std::size_t connectivity_size(regular_connectivity const &conn) { return conn.size_; }

        GT_FUNCTION friend std::integral_constant<std::size_t, MaxNeighbors> connectivity_max_neighbors(
            regular_connectivity const &) {
            return {};
        }

        GT_FUNCTION friend int connectivity_skip_value(regular_connectivity const &conn) { return conn.missing_value_; }

        friend auto connectivity_neighbor_table(regular_connectivity const &conn) {
            return gridtools::sid::rename_numbered_dimensions<LocationType, neighbor>(conn.tbl_);
        }
    };

} // namespace gridtools::next::atlas_wrappers

namespace atlas {
    template <template <class...> class L>
    decltype(auto) mesh_connectivity(L<vertex, edge>, const Mesh &mesh) {
        return gridtools::next::atlas_wrappers::regular_connectivity<vertex, 7
            // TODO this number must passed by the user (probably wrap atlas mesh)
            >{mesh.nodes().edge_connectivity()};
    }

    template <template <class...> class L>
    decltype(auto) mesh_connectivity(L<edge, vertex>, Mesh const &mesh) {
        return gridtools::next::atlas_wrappers::regular_connectivity<edge, 2>{mesh.edges().node_connectivity()};
    }

    template <template <class...> class L>
    decltype(auto) mesh_connectivity(L<edge>, Mesh const &mesh) {
        return gridtools::next::atlas_wrappers::primary_connectivity<edge>{std::size_t(mesh.edges().size())};
    }

    template <template <class...> class L>
    decltype(auto) mesh_connectivity(L<vertex>, Mesh const &mesh) {
        return gridtools::next::atlas_wrappers::primary_connectivity<vertex>{std::size_t(mesh.nodes().size())};
    }
} // namespace atlas
