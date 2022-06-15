#include <gtest/gtest.h>

#include GENERATED_FILE

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr inline auto in_elem = [](int node, int level, int elem = 0) {
  return node + level % (10 + elem);
};
constexpr inline auto in_ = [](int node, int level) {
  return tuple(in_elem(node, level, 0), in_elem(node, level, 1));
};

GT_REGRESSION_TEST(slice_sparse, test_environment<>, fn_backend_t) {

  using float_t = typename TypeParam::float_t;
  auto mesh = TypeParam::fn_unstructured_mesh();

  int nodes = mesh.nvertices();
  int levels = mesh.nlevels();

  auto actual = mesh.template make_storage<float_t>(nodes, levels);
  auto in = mesh.template make_const_storage<tuple<float_t, float_t>>(
      in_, nodes, levels);
  auto comp = [&] {
    generated::slice_sparse_fencil(
        fn_backend_t{}, unstructured_domain({nodes, levels}, {}), in, actual);
  };
  comp();

  TypeParam::verify(in_elem, actual);
}
} // namespace
