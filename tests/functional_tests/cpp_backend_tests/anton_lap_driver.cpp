#include <gtest/gtest.h>
#include <iostream>

#include GENERATED_FILE
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/fn/backend2/naive.hpp>
#include <gridtools/sid/sid_shift_origin.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

TEST(fn_lap, fn_backend_t) {
  double actual[10][10][3] = {};
  double in[10][10][3];
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      for (int k = 0; k < 3; ++k)
        in[i][j][k] = std::rand();

  auto expected = [&](auto i, auto j, auto k) {
    return in[i + 1][j][k] + in[i - 1][j][k] + in[i][j + 1][k] +
           in[i][j - 1][k] - 4 * in[i][j][k];
  };

  auto origin_shift =
      gridtools::hymap::keys<cartesian::dim::i, cartesian::dim::j>::values<
          gridtools::integral_constant<int, 1>,
          gridtools::integral_constant<int, 1>>{};

  auto shifted_in = sid::shift_sid_origin(in, origin_shift);
  auto shifted_actual = sid::shift_sid_origin(actual, origin_shift);

  auto domain = cartesian_domain(std::tuple{8, 8, 3});

  generated::lap_fencil(backend::naive{}, domain, shifted_actual, shifted_in);

  for (int i = 1; i < 9; ++i)
    for (int j = 1; j < 9; ++j)
      for (int k = 0; k < 3; ++k)
        EXPECT_DOUBLE_EQ(actual[i][j][k], expected(i, j, k))
            << "i=" << i << ", j=" << j << ", k=" << k;
}

} // namespace
