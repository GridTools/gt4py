/**
 * Simple 3x3 Cartesian, periodic, hand-made mesh.
 *
 *      0e    1e    2e
 *   | ---- | ---- | ---- |
 * 9e|0v 10e|1v 11e|2v  9e|0v
 *   |  0c  |  1c  |  2c  |
 *   |  3e  |  4e  |  5e  |
 *   | ---- | ---- | ---- |
 *12e|3v 13e|4v 14e|5v 12e|3v
 *   |  3c  |  4c  |  5c  |
 *   |  6e  |  7e  |  8e  |
 *   | ---- | ---- | ---- |
 *15e|6v 16e|7v 17e|8v 15e| 6v
 *   |  6c  |  7c  |  8c  |
 *   |  0e  |  1e  |  2e  |
 *   | ---- | ---- | ---- |
 *    0v     1v     2v     0v
 *
 */

#include "../mesh.hpp"
#include "../unstructured.hpp"
#include <cstddef>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>
#include <type_traits>

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace gridtools {
    namespace next {
        namespace test_helper {
            struct simple_mesh {
                template <class LocationType>
                struct primary_connectivity {
                    std::size_t size_;

                    GT_FUNCTION friend std::size_t connectivity_size(primary_connectivity const &conn) {
                        return conn.size_;
                    }
                };

                template <class LocationType, std::size_t MaxNeighbors>
                struct regular_connectivity {
                    struct builder {
                        auto operator()(std::size_t size) {
                            return gridtools::storage::builder<storage_trait>.template type<int>().template layout<0, 1>().template dimensions(
                    size, std::integral_constant<std::size_t, MaxNeighbors>{});
                        }
                    };

                    decltype(builder{}(0)()) tbl_;
                    std::size_t size_;
                    static constexpr gridtools::int_t missing_value_ = -1;

                    regular_connectivity(std::vector<std::array<int, MaxNeighbors>> tbl)
                        : tbl_{builder{}(tbl.size()).initializer([&tbl](std::size_t primary, std::size_t neigh) {
                              return tbl[primary][neigh];
                          })()},
                          size_{tbl.size()} {}

                    GT_FUNCTION friend std::size_t connectivity_size(regular_connectivity const &conn) {
                        return conn.size_;
                    }

                    GT_FUNCTION friend std::integral_constant<std::size_t, MaxNeighbors> connectivity_max_neighbors(
                        regular_connectivity const &) {
                        return {};
                    }

                    GT_FUNCTION friend int connectivity_skip_value(regular_connectivity const &conn) {
                        return conn.missing_value_;
                    }

                    friend auto connectivity_neighbor_table(regular_connectivity const &conn) {
                        return gridtools::sid::rename_numbered_dimensions<LocationType, neighbor>(conn.tbl_);
                    }
                };

                template <template <class...> class L>
                friend decltype(auto) mesh_connectivity(L<vertex>, simple_mesh const &) {
                    return primary_connectivity<vertex>{9};
                }
                template <template <class...> class L>
                friend decltype(auto) mesh_connectivity(L<edge>, simple_mesh const &) {
                    return primary_connectivity<edge>{18};
                }
                template <template <class...> class L>
                friend decltype(auto) mesh_connectivity(L<cell>, simple_mesh const &) {
                    return primary_connectivity<cell>{9};
                }
                template <template <class...> class L>
                friend decltype(auto) mesh_connectivity(L<cell, cell>, simple_mesh const &) {
                    return regular_connectivity<cell, 4>{{
                        {6, 1, 3, 2}, // 0
                        {7, 2, 4, 0}, // 1
                        {8, 0, 5, 1}, // 2
                        {0, 4, 6, 5}, // 3
                        {1, 5, 7, 3}, // 4
                        {2, 3, 8, 4}, // 5
                        {3, 7, 0, 8}, // 6
                        {4, 8, 1, 6}, // 7
                        {5, 6, 2, 7}  // 8
                    }};
                }
                template <template <class...> class L>
                friend decltype(auto) mesh_connectivity(L<edge, vertex>, simple_mesh const &) {
                    return regular_connectivity<edge, 2>{{
                        {0, 1}, // 0
                        {1, 2},
                        {2, 0},
                        {3, 4},
                        {4, 5},
                        {5, 3},
                        {6, 7},
                        {7, 8},
                        {8, 6},
                        {0, 3}, // 9
                        {1, 4},
                        {2, 5},
                        {3, 6},
                        {4, 7},
                        {5, 8},
                        {6, 0},
                        {7, 1},
                        {8, 2},
                    }};
                }
            };
        } // namespace test_helper
    }     // namespace next
} // namespace gridtools

/**
 * TODO maybe later: Simple 2x2 Cartesian hand-made mesh, non periodic, one halo line.
 *
 *     0e    1e    2e    3e
 *   | --- | --- | --- | --- |
 *   |0v   |1v   |2v   |3v   |4v
 *   |  0c |  1c |  2c |  3c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   |  4c |  5c |  6c |  7c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   |  8c |  9c | 10c | 11c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   | 12c | 13c | 14c | 15c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *
 *
 */
