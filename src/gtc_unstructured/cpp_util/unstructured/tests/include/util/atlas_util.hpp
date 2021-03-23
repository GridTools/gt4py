#pragma once

#include <atlas/functionspace/EdgeColumns.h>
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/Mesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/meshgenerator/MeshGenerator.h>

namespace atlas_util {

    auto inline make_mesh(std::string const &grid_type = "O2") {
        atlas::StructuredGrid structuredGrid = atlas::Grid(grid_type);
        return atlas::StructuredMeshGenerator{}.generate(structuredGrid);
    }

    template <typename T = double>
    auto inline make_edge_field(atlas::Mesh &mesh, int nb_levels) {
        atlas::mesh::actions::build_edges(mesh);
        atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));
        return fs_edges.createField<T>();
    }
} // namespace atlas_util
