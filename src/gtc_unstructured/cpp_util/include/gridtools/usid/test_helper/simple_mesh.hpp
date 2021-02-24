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

namespace gridtools::usid::test_helper {
    namespace impl_ {

        template <std::size_t MaxNeighbors>
        auto make_connectivity_producer(std::vector<std::array<int, MaxNeighbors>> const &tbl) {
            return [tbl](auto traits) {
                return gridtools::storage::builder<decltype(traits)>
                                .template type<int>()
                                .dimensions(tbl.size(), std::integral_constant<std::size_t,MaxNeighbors>{})
                                .initializer([&tbl](std::size_t p, std::size_t n){
                                    return tbl[p][n];})
                                .build();
            };
        }

    } // namespace impl_
    struct simple_mesh {
        static constexpr std::size_t vertices = 9;
        static constexpr std::size_t edges = 18;
        static constexpr std::size_t cells = 9;

        auto e2v() {
            return impl_::make_connectivity_producer<2>({//
                {0, 1},                                  // 0
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
                {8, 2}});
        }

        auto c2c() {
            // todo: fix this is c2e2c
            return impl_::make_connectivity_producer<4>({
                {6, 1, 3, 2}, // 0
                {7, 2, 4, 0}, // 1
                {8, 0, 5, 1}, // 2
                {0, 4, 6, 5}, // 3
                {1, 5, 7, 3}, // 4
                {2, 3, 8, 4}, // 5
                {3, 7, 0, 8}, // 6
                {4, 8, 1, 6}, // 7
                {5, 6, 2, 7}  // 8
            });
        }
    };
} // namespace gridtools::usid::test_helper

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
