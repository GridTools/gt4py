#include "../generated_vertex2edge_gtir_unaive.hpp"
#include <cpp_bindgen/export.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/storage/adapter/fortran_array_view.hpp>
#include <gridtools/usid/icon.hpp>
#include <gridtools/usid/test_helper/field_builder.hpp>
#include <gridtools/usid/test_helper/simple_mesh.hpp>

// #include <gtest/gtest.h>
#include <tuple>

namespace icon {
using namespace gridtools;

auto alloc_stencil_impl(int n_edges, fortran_array_view<int, 2> tbl) {
  return sten({-1, n_edges, -1, 1},
              make_connectivity_producer(integral_constant<int, 2>{}, tbl));
}

using stencil_t =
    decltype(alloc_stencil_impl(0, fortran_array_view<int, 2>{{}}));

template <class T>
void run_stencil_impl(stencil_t stencil, fortran_array_view<T, 1> in,
                      fortran_array_view<T, 1> out) {
  stencil(in, out);
}

BINDGEN_EXPORT_BINDING_WRAPPED(2, alloc_stencil, alloc_stencil_impl);
BINDGEN_EXPORT_GENERIC_BINDING_WRAPPED(3, run_stencil, run_stencil_impl,
                                       (double)(float));

} // namespace icon
