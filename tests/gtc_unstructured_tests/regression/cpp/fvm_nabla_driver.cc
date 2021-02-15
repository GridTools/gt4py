#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include <gtest/gtest.h>

#include <atlas/array.h>
#include <atlas/functionspace.h>
#include <atlas/grid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/actions/BuildDualMesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/meshgenerator.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include <gridtools/usid/atlas.hpp>

#ifdef __CUDACC__
#include <gridtools/storage/gpu.hpp>
namespace fvm_nabla_driver_impl_ {
    using storage_traits_t = gridtools::storage::gpu;
}
#else
#include <gridtools/storage/cpu_ifirst.hpp>
namespace fvm_nabla_driver_impl_ {
    using storage_traits_t = gridtools::storage::cpu_ifirst;
}
#endif

#include "${STENCIL_IMPL_SOURCE}"

namespace fvm_nabla_driver_impl_ {
    using namespace gridtools::literals;

    inline constexpr auto storage_builder = gridtools::storage::builder<storage_traits_t>;

    inline auto make_mesh() {
        using namespace atlas;
        using param_t = StructuredMeshGenerator::Parameters;
        auto res = StructuredMeshGenerator(param_t("triangulate", true) | param_t("angle", -1.)).generate(Grid("O32"));
        functionspace::EdgeColumns(res, option::levels(1) | option::halo(1));
        functionspace::NodeColumns(res, option::levels(1) | option::halo(1));
        mesh::actions::build_edges(res);
        mesh::actions::build_node_to_edge_connectivity(res);
        mesh::actions::build_median_dual_mesh(res);
        return res;
    }

    inline const double rpi = 2 * std::asin(1);
    inline const double radius = 6371.22e+03;
    inline const double deg2rad = 2 * rpi / 360;
    inline constexpr int MXX = 0;
    inline constexpr int MYY = 1;
    inline constexpr auto edges_per_node = 7_c;

    inline auto make_vol(atlas::mesh::Nodes const &nodes) {
        auto dual_volumes = atlas::array::make_view<double, 1>(nodes.field("dual_volumes"));
        auto init = [&](int n) { return dual_volumes(n) * (std::pow(deg2rad, 2) * std::pow(radius, 2)); };
        return storage_builder.type<double>().dimensions(nodes.size()).initializer(init).build();
    }

    inline auto make_sign(atlas::Mesh const &mesh) {
        auto &&edges = mesh.edges();
        auto &&nodes = mesh.nodes();
        auto &&n2e = nodes.edge_connectivity();
        auto &&e2n = edges.node_connectivity();
        auto flags = atlas::array::make_view<int, 1>(edges.flags());
        auto is_pole_edge = [&](auto e) {
            using topology_t = atlas::mesh::Nodes::Topology;
            return topology_t::check(flags(e), topology_t::POLE);
        };
        auto init = [&](int n, int, int e) -> double {
            if (e >= n2e.cols(n))
                return 0;
            auto ee = n2e(n, e);
            return n == e2n(ee, 0) || is_pole_edge(ee) ? 1 : -1;
        };
        return storage_builder.type<double>()
            .selector<1, 0, 1>()
            .dimensions(nodes.size(), 1, edges_per_node)
            .initializer(init)
            .build();
    }

    inline auto make_S_MXX(atlas::mesh::Edges const &edges) {
        auto dual_normals = atlas::array::make_view<double, 2>(edges.field("dual_normals"));
        return storage_builder.type<double>()
            .dimensions(edges.size())
            .initializer([&](int i) { return dual_normals(i, MXX) * radius * deg2rad; })
            .build();
    }

    inline auto make_S_MYY(atlas::mesh::Edges const &edges) {
        auto dual_normals = atlas::array::make_view<double, 2>(edges.field("dual_normals"));
        return storage_builder.type<double>()
            .dimensions(edges.size())
            .initializer([&](int i) { return dual_normals(i, MYY) * radius * deg2rad; })
            .build();
    }

    // TODO ask Christian for a proper name for this input data
    inline auto make_pp(atlas::mesh::Nodes const &nodes) {
        static const double zh0 = 2000;
        static const double zrad = 3 * rpi / 4 * radius;
        static const double zeta = rpi / 16 * radius;
        static const double zlatc = 0;
        static const double zlonc = 3 * rpi / 2;

        auto lonlat = atlas::array::make_view<double, 2>(nodes.field("lonlat"));
        // lonlatcr is in physical space and may differ from coords later
        auto rlonlatcr = [&](int n, int i) { return lonlat(n, i) * deg2rad; };
        auto init = [&](int n) {
            double zlon = rlonlatcr(n, MXX);
            double rcosa = std::cos(rlonlatcr(n, MYY));
            double rsina = std::sin(rlonlatcr(n, MYY));
            double zdist = std::sin(zlatc) * rsina + std::cos(zlatc) * rcosa * std::cos(zlon - zlonc);
            zdist = radius * std::acos(zdist);
            return zdist < zrad
                       ? .5 * zh0 * (1 + std::cos(rpi * zdist / zrad)) * std::pow(std::cos(rpi * zdist / zeta), 2)
                       : 0;
        };
        return storage_builder.type<double>().dimensions(nodes.size()).initializer(init).build();
    }

    template <class T>
    auto min_max(T const &field) {
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::min();
        auto view = field->const_host_view();
        auto lengths = field->lengths();
        for (int i = 0; i < (int)lengths[0]; ++i)
            for (int k = 0; k < (int)lengths[1]; ++k) {
                double x = view(i, k);
                min = std::min(min, x);
                max = std::max(max, x);
            }
        return std::make_tuple(min, max);
    }

    template <class Nabla>
    void fvm_nabla_driver(Nabla nabla) {
        constexpr auto k = 1_c;

        auto mesh = make_mesh();
        auto &&edges = mesh.edges();
        auto &&nodes = mesh.nodes();

        // output
        auto make_output = storage_builder.type<double>().dimensions(nodes.size(), k);
        auto pnabla_MXX = make_output();
        auto pnabla_MYY = make_output();

        nabla({nodes.size(), edges.size(), mesh.cells().size(), k},
            make_storage_producer(edges_per_node, nodes.edge_connectivity()),
            make_storage_producer(2_c, edges.node_connectivity()))(make_S_MXX(edges),
            make_S_MYY(edges),
            make_pp(nodes),
            pnabla_MXX,
            pnabla_MYY,
            make_vol(nodes),
            make_sign(mesh));

        auto [x_min, x_max] = min_max(pnabla_MXX);
        auto [y_min, y_max] = min_max(pnabla_MYY);

        EXPECT_DOUBLE_EQ(-3.5455427772566003E-003, x_min);
        EXPECT_DOUBLE_EQ(3.5455427772565435E-003, x_max);
        EXPECT_DOUBLE_EQ(-3.3540113705465301E-003, y_min);
        EXPECT_DOUBLE_EQ(3.3540113705465301E-003, y_max);
    } //
} // namespace fvm_nabla_driver_impl_
using fvm_nabla_driver_impl_::fvm_nabla_driver;

TEST(fvm, nabla_naive) { fvm_nabla_driver(nabla); }