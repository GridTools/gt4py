#include "atlas/grid.h"
#include "atlas/mesh/actions/BuildCellCentres.h"
#include "atlas/mesh/actions/BuildDualMesh.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"
#include "gridtools/common/integral_constant.hpp"
#include <array_fwd.h>
#include <atlas/array.h>
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <atlas/option.h>
#include <field/Field.h>
#include <functionspace/EdgeColumns.h>
#include <type_traits>

#include "gridtools/next/atlas_array_view_adapter.hpp"
#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/atlas_field_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/sid/synthetic.hpp>

#include "tests/include/util/atlas_util.hpp"

namespace dim {
    struct k;
} // namespace dim

template <class Ptr, class Strides, class UpperBounds>
__global__ void kernel(Ptr ptr_holder, Strides strides, UpperBounds upper_bounds) {
    auto ptr = ptr_holder();
    gridtools::sid::shift(ptr, gridtools::device::at_key<edge>(strides), threadIdx.x);
    for (int i = 0; i < gridtools::device::at_key<dim::k>(upper_bounds); ++i) {
        printf("%f\n", *ptr);
        gridtools::sid::shift(ptr, gridtools::device::at_key<dim::k>(strides), 1);
    }
}

int main() {
    auto mesh = atlas_util::make_mesh();
    atlas::mesh::actions::build_edges(mesh);

    int nb_levels = 5;
    atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    atlas::Field f;
    auto my_field = fs_edges.createField<double>(atlas::option::name("my_field"));

    auto view = atlas::array::make_view<double, 2>(my_field);
    for (int i = 0; i < fs_edges.size(); ++i)
        for (int k = 0; k < nb_levels; ++k)
            view(i, k) = i * 10 + k;

    auto my_field_as_data_store =
        gridtools::next::atlas_util::as_data_store<edge, dim::k>::with_type<double>{}(my_field);
    static_assert(gridtools::is_sid<decltype(my_field_as_data_store)>{});

    kernel<<<1, fs_edges.size()>>>(gridtools::sid::get_origin(my_field_as_data_store),
        gridtools::sid::get_strides(my_field_as_data_store),
        gridtools::sid::get_upper_bounds(my_field_as_data_store));
    cudaDeviceSynchronize();
}
