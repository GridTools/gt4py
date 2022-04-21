#include GENERATED_FILE

#include <iostream>

#include <gtest/gtest.h>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr inline auto in = [](auto... indices) { return (... + indices); };

GT_REGRESSION_TEST(fn_lap, test_environment<1>, fn_backend_t) {
  auto actual = TypeParam::make_storage();

  auto expected = [&](auto i, auto j, auto k) {
    return in(i + 1, j, k) + in(i - 1, j, k) + in(i, j + 1, k) +
           in(i, j - 1, k) - 4 * in(i, j, k);
  };

  generated::lap_fencil(
      fn_backend_t(),
      cartesian_domain(TypeParam::fn_cartesian_sizes(), std::tuple{1, 1, 0}),
      actual, TypeParam::make_const_storage(in));

  TypeParam::verify(expected, actual);
}
} // namespace
