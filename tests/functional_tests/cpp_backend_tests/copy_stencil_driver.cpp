#include <gtest/gtest.h>

#include GENERATED_FILE

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr inline auto in = [](auto... indices) { return (... + indices); };

GT_REGRESSION_TEST(fn_cartesian_copy, test_environment<>, fn_backend_t) {
  auto out = TypeParam::make_storage();

  auto comp = [&, in = TypeParam::make_const_storage(in)] {
    generated::copy_fencil(fn_backend_t{},
                           cartesian_domain(TypeParam::fn_cartesian_sizes()),
                           in, out);
  };
  comp();

  TypeParam::verify(in, out);
}

} // namespace
