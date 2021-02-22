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
  return sten({-1, n_edges, -1, 1}, make_connectivity_producer<2>(tbl));
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

// TEST(regression, vertex2edge_fortran) {
//   // auto in = test_helper::make_field<double>(simple_mesh::vertices);

//   double in[simple_mesh::vertices];
//   // auto view = in->host_view();
//   //  1   1   1 (1)
//   //  1   2   1 (1)
//   //  1   1   1 (1)
//   // (1) (1) (1)
//   for (std::size_t i = 0; i < 9; ++i)
//     in[i] = 1;
//   in[4] = 2;

//   auto in_fav = fortran_array_view<double,
//   1>{bindgen_fortran_array_descriptor{
//       bindgen_fortran_array_kind::bindgen_fk_Double,
//       1,
//       {simple_mesh::vertices},
//       &in,
//       false}};

//   // auto out = test_helper::make_field<double>(simple_mesh::edges);
//   double out[simple_mesh::edges];
//   auto out_fav = fortran_array_view<double,
//   1>{bindgen_fortran_array_descriptor{
//       bindgen_fortran_array_kind::bindgen_fk_Double,
//       1,
//       {simple_mesh::edges},
//       &out,
//       false}};

//   // sten({-1, simple_mesh::edges, -1, 1}, simple_mesh{}.e2v())(in, out);
//   run_stencil<double>(in_fav, out_fav);

//   // x 2 x 2 x 2
//   // 2   3   2
//   // x 3 x 3 x 2
//   // 2   3   2
//   // x 2 x 2 x 2
//   // 2   2   2
//   // auto out_view = out->const_host_view();
//   EXPECT_DOUBLE_EQ(2, out[0]);
//   EXPECT_DOUBLE_EQ(2, out[1]);
//   EXPECT_DOUBLE_EQ(2, out[2]);
//   EXPECT_DOUBLE_EQ(3, out[3]);
//   EXPECT_DOUBLE_EQ(3, out[4]);
//   EXPECT_DOUBLE_EQ(2, out[5]);
//   EXPECT_DOUBLE_EQ(2, out[6]);
//   EXPECT_DOUBLE_EQ(2, out[7]);
//   EXPECT_DOUBLE_EQ(2, out[8]);
//   EXPECT_DOUBLE_EQ(2, out[9]);
//   EXPECT_DOUBLE_EQ(3, out[10]);
//   EXPECT_DOUBLE_EQ(2, out[11]);
//   EXPECT_DOUBLE_EQ(2, out[12]);
//   EXPECT_DOUBLE_EQ(3, out[13]);
//   EXPECT_DOUBLE_EQ(2, out[14]);
//   EXPECT_DOUBLE_EQ(2, out[15]);
//   EXPECT_DOUBLE_EQ(2, out[16]);
//   EXPECT_DOUBLE_EQ(2, out[17]);
// }
} // namespace icon
