#include <gtest/gtest.h>

#include <fn_select.hpp>
#include <test_environment.hpp>
#include GENERATED_FILE

#include <gridtools/sid/rename_dimensions.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr inline auto in = [](auto... indices) { return (... + indices); };

GT_REGRESSION_TEST(fn_cartesian_copy, test_environment<>, fn_backend_t) {
  auto out = TypeParam::make_storage();
  auto out_wrapped =
      sid::rename_numbered_dimensions<generated::IDim_t, generated::JDim_t,
                                      generated::KDim_t>(out);
  auto out2 = TypeParam::make_storage();
  auto out2_wrapped =
      sid::rename_numbered_dimensions<generated::IDim_t, generated::JDim_t,
                                      generated::KDim_t>(out2);

  auto in_wrapped =
      sid::rename_numbered_dimensions<generated::IDim_t, generated::JDim_t,
                                      generated::KDim_t>(
          TypeParam::make_const_storage(in));
  auto comp = [&] {
    generated::copy_program(tuple{})(
        fn_backend_t{}, in_wrapped, out_wrapped, out2_wrapped,
        at_key<cartesian::dim::i>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::j>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::k>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::i>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::j>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::k>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::i>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::j>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::k>(TypeParam::fn_cartesian_sizes()));
  };
  comp();

  TypeParam::verify(in, out);
  TypeParam::verify(in, out2);
}

} // namespace
