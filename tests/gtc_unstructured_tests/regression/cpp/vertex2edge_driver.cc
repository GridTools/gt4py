#include "${STENCIL_IMPL_SOURCE}"
#include <gridtools/next/test_helper/field_builder.hpp>
#include <gridtools/next/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>
#include <tuple>

namespace {

using namespace gridtools::next;

TEST(regression, vertex2edge) {
  test_helper::simple_mesh mesh;

  auto e2v = mesh::connectivity<std::tuple<edge, vertex>>(mesh);

  auto in = test_helper::make_field<double, vertex>(mesh);

  // TODO discuss with anstaf what an unstructured field should be, here I steal
  // the data_store from a SID
  auto view = in.m_impl->host_view();
  //  1   1   1 (1)
  //  1   2   1 (1)
  //  1   1   1 (1)
  // (1) (1) (1)
  for (std::size_t i = 0; i < 9; ++i)
    view(i) = 1;
  view(4) = 2;

  auto out = test_helper::make_field<double, edge>(mesh);
  sten(mesh, in, out);

  // x 2 x 2 x 2
  // 2   3   2
  // x 3 x 3 x 2
  // 2   3   2
  // x 2 x 2 x 2
  // 2   2   2
  auto out_view = out.m_impl->const_host_view();
  EXPECT_DOUBLE_EQ(2, out_view(0));
  EXPECT_DOUBLE_EQ(2, out_view(1));
  EXPECT_DOUBLE_EQ(2, out_view(2));
  EXPECT_DOUBLE_EQ(3, out_view(3));
  EXPECT_DOUBLE_EQ(3, out_view(4));
  EXPECT_DOUBLE_EQ(2, out_view(5));
  EXPECT_DOUBLE_EQ(2, out_view(6));
  EXPECT_DOUBLE_EQ(2, out_view(7));
  EXPECT_DOUBLE_EQ(2, out_view(8));
  EXPECT_DOUBLE_EQ(2, out_view(9));
  EXPECT_DOUBLE_EQ(3, out_view(10));
  EXPECT_DOUBLE_EQ(2, out_view(11));
  EXPECT_DOUBLE_EQ(2, out_view(12));
  EXPECT_DOUBLE_EQ(3, out_view(13));
  EXPECT_DOUBLE_EQ(2, out_view(14));
  EXPECT_DOUBLE_EQ(2, out_view(15));
  EXPECT_DOUBLE_EQ(2, out_view(16));
  EXPECT_DOUBLE_EQ(2, out_view(17));
}
} // namespace
