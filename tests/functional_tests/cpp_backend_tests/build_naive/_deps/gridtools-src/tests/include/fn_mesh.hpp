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

#include <cassert>

#include <gridtools/storage/builder.hpp>
#include <type_traits>

namespace gridtools {

    struct v2e {};
    struct e2v {};

    template <class StorageTraits, class FloatType>
    class structured_unstructured_mesh {
        int m_nx, m_ny, m_nz;

        constexpr auto v2e_initializer() const {
            return [nx = m_nx, ny = m_ny](int vertex) {
                assert(vertex >= 0 && vertex < nx * ny);
                int const nxedges = (nx - 1) * ny;
                int const nyedges = nx * (ny - 1);
                int i = vertex % nx;
                int j = vertex / nx;
                array<int, max_v2e_neighbors_t::value> neighbors;
                int n = 0;
                if (i > 0)
                    neighbors[n++] = (i - 1) + (nx - 1) * j;
                if (i < nx - 1)
                    neighbors[n++] = i + (nx - 1) * j;
                if (j > 0)
                    neighbors[n++] = nxedges + i + nx * (j - 1);
                if (j < ny - 1)
                    neighbors[n++] = nxedges + i + nx * j;
                if (i < nx - 1 && j > 0)
                    neighbors[n++] = nxedges + nyedges + i + (nx - 1) * (j - 1);
                if (i > 0 && j < ny - 1)
                    neighbors[n++] = nxedges + nyedges + (i - 1) + (nx - 1) * j;
                for (; n < neighbors.size(); ++n)
                    neighbors[n] = -1;
                return neighbors;
            };
        }

        constexpr auto e2v_initializer() const {
            return [nx = m_nx, ny = m_ny](int edge) {
                int const nxedges = (nx - 1) * ny;
                int const nyedges = nx * (ny - 1);
                [[maybe_unused]] int const nxyedges = (nx - 1) * (ny - 1);
                assert(edge >= 0 && edge < nxedges + nyedges + nxyedges);
                if (edge < nxedges) {
                    int i = edge % (nx - 1);
                    int j = edge / (nx - 1);
                    return array{i + nx * j, i + 1 + nx * j};
                }
                edge -= nxedges;
                if (edge < nyedges) {
                    int i = edge % nx;
                    int j = edge / nx;
                    return array{i + nx * j, i + nx * (j + 1)};
                }
                edge -= nyedges;
                assert(edge < nxyedges);
                int i = edge % (nx - 1);
                int j = edge / (nx - 1);
                return array{i + 1 + nx * j, i + nx * (j + 1)};
            };
        }

      public:
        using max_v2e_neighbors_t = std::integral_constant<int, 6>;
        using max_e2v_neighbors_t = std::integral_constant<int, 2>;

        constexpr structured_unstructured_mesh(int nx, int ny, int nz) : m_nx(nx), m_ny(ny), m_nz(nz) {}

        constexpr int nvertices() const { return m_nx * m_ny; }
        constexpr int nedges() const {
            int nxedges = (m_nx - 1) * m_ny;
            int nyedges = m_nx * (m_ny - 1);
            int nxyedges = (m_nx - 1) * (m_ny - 1);
            return nxedges + nyedges + nxyedges;
        }
        constexpr int nlevels() const { return m_nz; }

        template <class T = FloatType, class Init, class... Dims, std::enable_if_t<!std::is_integral_v<Init>, int> = 0>
        auto make_storage(Init const &init, Dims... dims) const {
            return storage::builder<StorageTraits>.dimensions(dims...).template type<T>().initializer(init).unknown_id().build();
        }

        template <class T = FloatType,
            class... Dims,
            std::enable_if_t<std::conjunction_v<std::is_integral<Dims>...>, int> = 0>
        auto make_storage(Dims... dims) const {
            return make_storage<T>([](int, int) { return T(); }, dims...);
        }

        template <class T = FloatType, class... Args>
        auto make_const_storage(Args &&... args) const {
            return make_storage<T const>(std::forward<Args>(args)...);
        }

        auto v2e_table() const {
            return storage::builder<StorageTraits>.dimensions(nvertices()).template type<array<int, max_v2e_neighbors_t::value>>().initializer(v2e_initializer()).unknown_id().build();
        }

        auto e2v_table() const {
            return storage::builder<StorageTraits>.dimensions(nedges()).template type<array<int, max_e2v_neighbors_t::value>>().initializer(e2v_initializer()).unknown_id().build();
        }
    };

} // namespace gridtools
