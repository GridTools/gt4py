#include "${STENCIL_IMPL_SOURCE}"
#include <gridtools/next/test_helper/field_builder.hpp>
#include <gridtools/next/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>
#include <tuple>

namespace {

using namespace gridtools::next;

TEST(regression, temporary) {
  test_helper::simple_mesh mesh;

  auto in = test_helper::make_field<double, cell>(mesh);

  // TODO discuss with anstaf what an unstructured field should be, here I steal
  // the data_store from a SID
  auto view = in.m_impl->host_view();
  //  1   2   3
  //  4   5   6
  //  7   8   9
  for (std::size_t i = 0; i < 9; ++i)
    view(i) = i;

  auto out = test_helper::make_field<double, cell>(mesh);
  sten(mesh, in, out);

  //  1   2   3
  //  4   5   6
  //  7   8   9
  auto out_view = out.m_impl->const_host_view();
  EXPECT_DOUBLE_EQ(0, out_view(0));
  EXPECT_DOUBLE_EQ(1, out_view(1));
  EXPECT_DOUBLE_EQ(2, out_view(2));
  EXPECT_DOUBLE_EQ(3, out_view(3));
  EXPECT_DOUBLE_EQ(4, out_view(4));
  EXPECT_DOUBLE_EQ(5, out_view(5));
  EXPECT_DOUBLE_EQ(6, out_view(6));
  EXPECT_DOUBLE_EQ(7, out_view(7));
  EXPECT_DOUBLE_EQ(8, out_view(8));
}
} // namespace
