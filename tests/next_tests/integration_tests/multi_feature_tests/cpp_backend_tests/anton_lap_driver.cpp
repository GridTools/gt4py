#include <gtest/gtest.h>

#include "fn_select_gt4py.hpp"
#include <test_environment.hpp>
#include GENERATED_FILE

#include <gridtools/sid/rename_dimensions.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr inline auto in = [](auto... indices) { return (... + indices); };

using backend_t =
    fn_backend_t<block_sizes_t<generated::IDim_t, generated::JDim_t,
                               generated::KDim_t>::values<32, 8, 1>>;

GT_REGRESSION_TEST(fn_lap, test_environment<1>, backend_t) {
  auto actual = TypeParam::make_storage();

  auto expected = [&](auto i, auto j, auto k) {
    return in(i + 1, j, k) + in(i - 1, j, k) + in(i, j + 1, k) +
           in(i, j - 1, k) - 4 * in(i, j, k);
  };

  generated::lap_fencil(tuple{})(
      backend_t(), at_key<cartesian::dim::i>(TypeParam::fn_cartesian_sizes()),
      at_key<cartesian::dim::j>(TypeParam::fn_cartesian_sizes()),
      at_key<cartesian::dim::k>(TypeParam::fn_cartesian_sizes()), 1, 1, 0,
      sid::rename_numbered_dimensions<generated::IDim_t, generated::JDim_t,
                                      generated::KDim_t>(actual),
      sid::rename_numbered_dimensions<generated::IDim_t, generated::JDim_t,
                                      generated::KDim_t>(
          TypeParam::make_const_storage(in)));

  TypeParam::verify(expected, actual);
}
} // namespace
