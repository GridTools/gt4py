#include <gtest/gtest.h>

#include GENERATED_FILE

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr inline auto a = [](auto...) { return -1; };
constexpr inline auto b = [](auto...) { return 3; };
constexpr inline auto c = [](auto...) { return 1; };
constexpr inline auto d = [](int ksize) {
  return [kmax = ksize - 1](auto... indices) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    GT_NVCC_DIAG_PUSH_SUPPRESS(174)
    int k = (..., indices);
#pragma GCC diagnostic pop
    GT_NVCC_DIAG_POP_SUPPRESS(174)
    return k == 0 ? 4 : k == kmax ? 2 : 3;
  };
};
constexpr inline auto expected = [](auto...) { return 1; };

GT_REGRESSION_TEST(fn_cartesian_tridiagonal_solve, vertical_test_environment<>,
                   fn_backend_t) {
  using float_t = typename TypeParam::float_t;

  auto x = TypeParam::make_storage();
  auto comp = [&, a = TypeParam::make_const_storage(a),
               b = TypeParam::make_const_storage(b),
               c = TypeParam::make_const_storage(c),
               d = TypeParam::make_const_storage(d(TypeParam::d(2)))] {
    generated::tridiagonal_solve_fencil(
        fn_backend_t(), cartesian_domain(TypeParam::fn_cartesian_sizes()), a, b,
        c, d, x);
  };
  comp();
  TypeParam::verify(expected, x);
}

} // namespace
