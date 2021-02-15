#include "${STENCIL_IMPL_SOURCE}"
#include <gridtools/usid/test_helper/field_builder.hpp>
#include <gridtools/usid/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>
#include <tuple>

namespace gridtools::usid {

using namespace gridtools::usid::test_helper;

TEST(regression, copy_with_k) {
  std::size_t k_size = 3;
  auto in = test_helper::make_field<double>(simple_mesh::edges, k_size);

  auto view = in->host_view();
  for (std::size_t i = 0; i < simple_mesh::edges; ++i)
    for (std::size_t k = 0; k < k_size; ++k)
      view(i, k) = i * k_size + k;

  auto out = test_helper::make_field<double>(simple_mesh::edges, k_size);

  sten({-1, simple_mesh::edges, -1, k_size})(in, out);

  // x 2 x 2 x 2
  // 2   3   2
  // x 3 x 3 x 2
  // 2   3   2
  // x 2 x 2 x 2
  // 2   2   2
  auto out_view = out->const_host_view();
  for (std::size_t i = 0; i < simple_mesh::edges; ++i)
    for (std::size_t k = 0; k < k_size; ++k)
      EXPECT_EQ(i * k_size + k, out_view(i, k)) << "at " << i << "/" << k;
}
} // namespace gridtools::usid
