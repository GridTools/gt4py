#include <fstream>
#include <limits>

#include "atlas/functionspace/EdgeColumns.h"
#include "atlas/functionspace/NodeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildDualMesh.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include "atlas/option/Options.h"
#include "atlas/output/Gmsh.h"
#include "gridtools/next/mesh.hpp"
#include <atlas/array.h>
#include <atlas/mesh.h>
#include <atlas/mesh/Nodes.h>

#include <gridtools/sid/composite.hpp>

#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/next/atlas_field_util.hpp>
#include <tuple>

#include <gtest/gtest.h>

#include "${STENCIL_IMPL_SOURCE}"

namespace {
    template <typename DS>
    std::tuple<double, double, double> min_max(DS const &field) {
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::min();
        auto view = field.m_impl->host_view(); // we are mixing SID with data_store concepts...

        auto first_length = field.m_impl->lengths()[0];

        double avg = 0.;
        for (std::size_t jnode = 0; jnode < first_length; ++jnode) {
            min = std::min(min, view(jnode, 0));
            max = std::max(max, view(jnode, 0));
            avg += view(jnode, 0);
        }
        return {min, max, avg / (double)first_length};
    }
} // namespace

namespace {
    const double rpi = 2.0 * std::asin(1.0);
    const double radius = 6371.22e+03;
    const double deg2rad = 2. * rpi / 360.;

    const int MXX = 0;
    const int MYY = 1;
} // namespace

class FVMDriver {
  private:
    atlas::Mesh mesh_;
    atlas::functionspace::EdgeColumns fs_edges_;
    atlas::functionspace::NodeColumns fs_nodes_;
    int nb_levels_;

    atlas::Field m_S_MXX;
    atlas::Field m_S_MYY;

    atlas::Field m_vol;
    atlas::Field m_sign;

  public:
    constexpr static int edges_per_node = 7;

    FVMDriver(std::string const &grid, int nb_levels)
        : mesh_{[&grid]() {
              atlas::StructuredGrid structuredGrid = atlas::Grid(grid);
              atlas::MeshGenerator::Parameters generatorParams;
              // generatorParams.set("three_dimensional", false);
              generatorParams.set("triangulate", true);
              // generatorParams.set("patch_pole", true);
              // generatorParams.set("include_pole", false);
              //   generatorParams.set("angle", 20);
              generatorParams.set("angle", -1.0);
              // generatorParams.set("ghost_at_end", true);

              atlas::StructuredMeshGenerator generator(generatorParams);
              return generator.generate(structuredGrid);
          }()},
          fs_edges_(mesh_, atlas::option::levels(nb_levels) | atlas::option::halo(1)),
          fs_nodes_(mesh_,
              atlas::option::levels(nb_levels) | atlas::option::halo(1)), //
          nb_levels_(nb_levels), m_S_MXX(fs_edges_.createField<double>(atlas::option::name("S_MXX"))),
          m_S_MYY(fs_edges_.createField<double>(atlas::option::name("S_MYY"))),
          m_vol(fs_nodes_.createField<double>(atlas::option::name("vol"))),
          m_sign(fs_nodes_.createField<double>(
              atlas::option::name("m_sign") | atlas::option::variables(FVMDriver::edges_per_node))) {
        atlas::mesh::actions::build_edges(mesh_);
        atlas::mesh::actions::build_node_to_edge_connectivity(mesh_);
        atlas::mesh::actions::build_median_dual_mesh(mesh_);

        initialize_S();
        // print_min_max(m_S_MXX);
        // print_min_max(m_S_MYY);
        initialize_sign();
        initialize_vol();
        // print_min_max(m_vol);
    }

  private:
    void initialize_vol() {
        // print_min_max_1d(mesh_.nodes().field("dual_volumes"));
        const auto vol_atlas = atlas::array::make_view<double, 1>(mesh_.nodes().field("dual_volumes"));
        auto vol = atlas::array::make_view<double, 2>(m_vol);
        for (int i = 0, size = vol_atlas.size(); i < size; ++i) {
            vol(i, 0) = vol_atlas(i) * (std::pow(deg2rad, 2) * std::pow(radius, 2));
        }
    }
    void initialize_S() {
        // all fields supported by dawn are 2 (or 3 with sparse) dimensional:
        // (unstructured, lev, sparse) S has dimensions (unstructured, [MMX/MMY])
        const auto S = atlas::array::make_view<double, 2>(mesh_.edges().field("dual_normals"));

        auto S_MXX = atlas::array::make_view<double, 2>(m_S_MXX);
        auto S_MYY = atlas::array::make_view<double, 2>(m_S_MYY);

        assert(nb_levels_ == 1);
        int klevel = 0;
        for (int i = 0, size = mesh_.edges().size(); i < size; ++i) {
            S_MXX(i, klevel) = S(i, MXX) * radius * deg2rad;
            S_MYY(i, klevel) = S(i, MYY) * radius * deg2rad;
        }
    }

    void initialize_sign() {
        auto node2edge_sign = atlas::array::make_view<double, 3>(m_sign);

        auto edge_flags = atlas::array::make_view<int, 1>(mesh_.edges().flags());
        using Topology = atlas::mesh::Nodes::Topology;
        auto is_pole_edge = [&](size_t e) { return Topology::check(edge_flags(e), Topology::POLE); };

        for (atlas::idx_t jnode = 0; jnode < mesh_.nodes().size(); ++jnode) {
            auto const &node_edge_connectivity = mesh_.nodes().edge_connectivity();
            auto const &edge_node_connectivity = mesh_.edges().node_connectivity();
            for (atlas::idx_t jedge = 0; jedge < node_edge_connectivity.cols(jnode); ++jedge) {
                auto iedge = node_edge_connectivity(jnode, jedge);
                auto ip1 = edge_node_connectivity(iedge, 0);
                if (jnode == ip1) {
                    node2edge_sign(jnode, 0, jedge) = 1.;
                } else {
                    node2edge_sign(jnode, 0, jedge) = -1.;
                    if (is_pole_edge(iedge)) {
                        node2edge_sign(jnode, 0, jedge) = 1.;
                    }
                }
            }
        }
    }

  public:
    atlas::Mesh const &mesh() const { return mesh_; }
    atlas::Mesh &mesh() { return mesh_; }
    atlas::functionspace::EdgeColumns const &fs_edges() const { return fs_edges_; }
    atlas::functionspace::NodeColumns const &fs_nodes() const { return fs_nodes_; }
    int nb_levels() const { return nb_levels_; }
    atlas::Field &S_MXX() { return m_S_MXX; }
    atlas::Field &S_MYY() { return m_S_MYY; }
    atlas::Field &vol() { return m_vol; }
    atlas::Field &sign() { return m_sign; }

    // TODO ask Christian for a proper name for this input data
    void fillInputData(atlas::Field &field) const {
        double zh0 = 2000.0;
        double zrad = 3. * rpi / 4.0 * radius;
        double zeta = rpi / 16.0 * radius;
        double zlatc = 0.0;
        double zlonc = 3.0 * rpi / 2.0;

        atlas::Field m_rlonlatcr{
            fs_nodes_.createField<double>(atlas::option::name("rlonlatcr") | atlas::option::variables(edges_per_node))};
        auto rlonlatcr = atlas::array::make_view<double, 3>(m_rlonlatcr);

        atlas::Field m_rcoords{
            fs_nodes_.createField<double>(atlas::option::name("rcoords") | atlas::option::variables(edges_per_node))};
        auto rcoords = atlas::array::make_view<double, 3>(m_rcoords);

        atlas::Field m_rcosa{fs_nodes_.createField<double>(atlas::option::name("rcosa"))};
        auto rcosa = atlas::array::make_view<double, 2>(m_rcosa);

        atlas::Field m_rsina{fs_nodes_.createField<double>(atlas::option::name("rsina"))};
        auto rsina = atlas::array::make_view<double, 2>(m_rsina);

        auto rzs = atlas::array::make_view<double, 2>(field);

        std::size_t k_level = 0;

        const auto rcoords_deg = atlas::array::make_view<double, 2>(mesh_.nodes().field("lonlat"));

        for (atlas::idx_t jnode = 0; jnode < mesh_.nodes().size(); ++jnode) {
            for (std::size_t i = 0; i < 2; ++i) {
                rcoords(jnode, k_level, i) = rcoords_deg(jnode, i) * deg2rad;
                rlonlatcr(jnode, k_level, i) = rcoords(jnode, k_level, i); // lonlatcr is in physical space and may
                                                                           // differ from coords later
            }
            rcosa(jnode, k_level) = cos(rlonlatcr(jnode, k_level, MYY));
            rsina(jnode, k_level) = sin(rlonlatcr(jnode, k_level, MYY));
        }
        for (atlas::idx_t jnode = 0; jnode < mesh_.nodes().size(); ++jnode) {
            double zlon = rlonlatcr(jnode, k_level, MXX);
            //   double zlat = rlonlatcr(jnode, k_level, MYY);
            double zdist = sin(zlatc) * rsina(jnode, k_level) + cos(zlatc) * rcosa(jnode, k_level) * cos(zlon - zlonc);
            zdist = radius * acos(zdist);
            rzs(jnode, k_level) = 0.0;
            if (zdist < zrad) {
                rzs(jnode, k_level) = rzs(jnode, k_level) + 0.5 * zh0 * (1.0 + cos(rpi * zdist / zrad)) *
                                                                std::pow(cos(rpi * zdist / zeta), 2);
            }
        }
    }
};

TEST(FVM, nabla) {

    FVMDriver driver{"O32", 1};
    // std::cout << "n_edges = " << driver.fs_edges()->size() << std::endl;
    // std::cout << "n_nodes = " << driver.fs_nodes()->size() << std::endl;

    // input
    atlas::Field m_pp = driver.fs_nodes().createField<double>(atlas::option::name("pp"));

    // output
    atlas::Field m_pnabla_MXX = driver.fs_nodes().createField<double>(atlas::option::name("pnabla_MXX"));
    atlas::Field m_pnabla_MYY = driver.fs_nodes().createField<double>(atlas::option::name("pnabla_MYY"));

    // temporary
    //  note: temporary fields are allocated in the stencil, but are still here if needed for debugging purposes
    //atlas::Field m_zavgS_MXX = driver.fs_edges().createField<double>(atlas::option::name("zavgS_MXX"));
    //atlas::Field m_zavgS_MYY = driver.fs_edges().createField<double>(atlas::option::name("zavgS_MYY"));

    driver.fillInputData(m_pp);
    //   print_min_max(m_pp);

    //   atlas::output::Gmsh gmesh("mymesh.msh");
    //   gmesh.write(driver.mesh());
    //   gmesh.write(m_pp);

    auto edge_sid = [](auto &&field) {
        return gridtools::next::atlas_util::as_data_store<edge, dim::k>::with_type<double>{}(field);
    };
    auto node_sid = [](auto &&field) {
        return gridtools::next::atlas_util::as_data_store<vertex, dim::k>::with_type<double>{}(field);
    };

    auto S_MXX_ds = edge_sid(driver.S_MXX());
    auto S_MYY_ds = edge_sid(driver.S_MYY());
    //auto m_zavgS_MXX_ds = edge_sid(m_zavgS_MXX);
    //auto m_zavgS_MYY_ds = edge_sid(m_zavgS_MYY);
    auto m_pp_ds = node_sid(m_pp);
    auto m_pnabla_MXX_ds = node_sid(m_pnabla_MXX);
    auto m_pnabla_MYY_ds = node_sid(m_pnabla_MYY);
    auto vol_ds = node_sid(driver.vol());
    auto sign_ds =
        gridtools::next::atlas_util::as_data_store<vertex, dim::k, neighbor>::with_type<double>{}(driver.sign());

    nabla(driver.mesh(),
        S_MXX_ds,
        S_MYY_ds,
        m_pp_ds,
        m_pnabla_MXX_ds,
        m_pnabla_MYY_ds,
        vol_ds,
        sign_ds);

    //   gmesh.write(m_pnabla_MXX);

    /*auto [zavgS_x_min, zavgS_x_max, zavgS_x_avg] = min_max(m_zavgS_MXX_ds);
    auto [zavgS_y_min, zavgS_y_max, zavgS_y_avg] = min_max(m_zavgS_MYY_ds);
    EXPECT_DOUBLE_EQ(-199755464.25741270, zavgS_x_min);
    EXPECT_DOUBLE_EQ(388241977.58389181, zavgS_x_max);
    EXPECT_DOUBLE_EQ(-1000788897.3202186, zavgS_y_min);
    EXPECT_DOUBLE_EQ(1000788897.3202186, zavgS_y_max);*/

    auto [x_min, x_max, x_avg] = min_max(m_pnabla_MXX_ds);
    auto [y_min, y_max, y_avg] = min_max(m_pnabla_MYY_ds);

    ASSERT_DOUBLE_EQ(-3.5455427772566003E-003, x_min);
    ASSERT_DOUBLE_EQ(3.5455427772565435E-003, x_max);
    ASSERT_DOUBLE_EQ(-3.3540113705465301E-003, y_min);
    ASSERT_DOUBLE_EQ(3.3540113705465301E-003, y_max);
}
