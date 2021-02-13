#include "${STENCIL_IMPL_SOURCE}"
#include <gridtools/usid/test_helper/field_builder.hpp>
#include <gridtools/usid/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>
#include <tuple>

namespace gridtools::usid {

using namespace gridtools::usid::test_helper;

TEST(regression, vertex2edge) {
  auto in = test_helper::make_field<double>(simple_mesh::vertices);

  auto view = in->host_view();
  //  1   1   1 (1)
  //  1   2   1 (1)
  //  1   1   1 (1)
  // (1) (1) (1)
  for (std::size_t i = 0; i < 9; ++i)
    view(i) = 1;
  view(4) = 2;

  auto out = test_helper::make_field<double>(simple_mesh::edges);

  sten({-1, simple_mesh::edges, -1, 1}, simple_mesh{}.e2v())(in, out);

  // x 2 x 2 x 2
  // 2   3   2
  // x 3 x 3 x 2
  // 2   3   2
  // x 2 x 2 x 2
  // 2   2   2
  auto out_view = out->const_host_view();
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
} // namespace gridtools::usid
