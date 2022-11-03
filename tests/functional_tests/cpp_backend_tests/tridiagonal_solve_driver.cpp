#include <gtest/gtest.h>

#include <fn_select.hpp>
#include <test_environment.hpp>
#include GENERATED_FILE

#include <gridtools/sid/rename_dimensions.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr inline auto a = [](auto...) { return -1; };
constexpr inline auto b = [](auto...) { return 3; };
constexpr inline auto c = [](auto...) { return 1; };
constexpr inline auto d = [](int ksize) {
  return [kmax = ksize - 1](auto, auto, auto k) {
    return k == 0 ? 4 : k == kmax ? 2 : 3;
  };
};
constexpr inline auto expected = [](auto...) { return 1; };

GT_REGRESSION_TEST(fn_cartesian_tridiagonal_solve, vertical_test_environment<>,
                   fn_backend_t) {
  using float_t = typename TypeParam::float_t;

  auto wrap = [](auto &&storage) {
    return sid::rename_numbered_dimensions<generated::IDim_t, generated::JDim_t,
                                           generated::KDim_t>(
        std::forward<decltype(storage)>(storage));
  };

  auto x = TypeParam::make_storage();
  auto x_wrapped = wrap(x);
  auto a_wrapped = wrap(TypeParam::make_const_storage(a));
  auto b_wrapped = wrap(TypeParam::make_const_storage(b));
  auto c_wrapped = wrap(TypeParam::make_const_storage(c));
  auto d_wrapped = wrap(TypeParam::make_const_storage(d(TypeParam::d(2))));
  auto comp = [&] {
    generated::tridiagonal_solve_fencil(tuple{})(
        fn_backend_t(),
        at_key<cartesian::dim::i>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::j>(TypeParam::fn_cartesian_sizes()),
        at_key<cartesian::dim::k>(TypeParam::fn_cartesian_sizes()), a_wrapped,
        b_wrapped, c_wrapped, d_wrapped, x_wrapped);
  };
  comp();
  TypeParam::verify(expected, x);
}

} // namespace
