#include <gtest/gtest.h>

#include <fn_select.hpp>
#include <test_environment.hpp>
#include GENERATED_FILE

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

// copied from gridtools::fn test
constexpr inline auto pp = [](int vertex, int k) { return (vertex + k) % 19; };
constexpr inline auto sign = [](int vertex) {
  return array<int, 6>{0, 1, vertex % 2, 1, (vertex + 1) % 2, 0};
};
constexpr inline auto vol = [](int vertex) { return vertex % 13 + 1; };
constexpr inline auto s = [](int edge, int k) {
  return tuple((edge + k) % 17, (edge + k) % 7);
};

constexpr inline auto zavg = [](auto const &e2v) {
  return [&e2v](int edge, int k) {
    double tmp = 0.0;
    for (int neighbor = 0; neighbor < 2; ++neighbor)
      tmp += pp(e2v(edge)[neighbor], k);
    tmp /= 2.0;
    return tuple{tmp * get<0>(s(edge, k)), tmp * get<1>(s(edge, k))};
  };
};

constexpr inline auto make_zavg_expected = [](auto const &mesh) {
  return [e2v_table = mesh.e2v_table()](int edge, int k) {
    auto e2v = e2v_table->const_host_view();
    return zavg(e2v)(edge, k);
  };
};

constexpr inline auto expected = [](auto const &v2e, auto const &e2v) {
  return [&v2e, zavg = zavg(e2v)](int vertex, int k) {
    auto res = tuple(0.0, 0.0);
    for (int neighbor = 0; neighbor < 6; ++neighbor) {
      int edge = v2e(vertex)[neighbor];
      if (edge != -1) {
        get<0>(res) += get<0>(zavg(edge, k)) * sign(vertex)[neighbor];
        get<1>(res) += get<1>(zavg(edge, k)) * sign(vertex)[neighbor];
      }
    }
    get<0>(res) /= vol(vertex);
    get<1>(res) /= vol(vertex);
    return res;
  };
};

constexpr inline auto make_expected = [](auto const &mesh) {
  return [v2e_table = mesh.v2e_table(),
          e2v_table = mesh.e2v_table()](int vertex, int k) {
    auto v2e = v2e_table->const_host_view();
    auto e2v = e2v_table->const_host_view();
    return expected(v2e, e2v)(vertex, k);
  };
};

// GT_REGRESSION_TEST(unstructured_zavg, test_environment<>, fn_backend_t) {
//   using float_t = typename TypeParam::float_t;

//   auto mesh = TypeParam::fn_unstructured_mesh();
//   auto actual = mesh.template make_storage<tuple<float_t, float_t>>(
//       mesh.nedges(), mesh.nlevels());

//   auto pp_ = mesh.make_const_storage(pp, mesh.nvertices(), mesh.nlevels());
//   auto s_ = mesh.template make_const_storage<tuple<float_t, float_t>>(
//       s, mesh.nedges(), mesh.nlevels());

//   auto e2v_conn =
//       connectivity<generated::E2V_t>(mesh.e2v_table()->get_const_target_ptr());
//   auto edge_domain =
//       unstructured_domain({mesh.nedges(), mesh.nlevels()}, {}, e2v_conn);

//   generated::zavgS_fencil(fn_backend_t{}, edge_domain, actual, pp_, s_);

//   auto expected = make_zavg_expected(mesh);
//   TypeParam::verify(expected, actual);
// }

GT_REGRESSION_TEST(unstructured_nabla, test_environment<>, fn_backend_t) {
  using float_t = typename TypeParam::float_t;

  auto mesh = TypeParam::fn_unstructured_mesh();
  auto actual = mesh.template make_storage<tuple<float_t, float_t>>(
      mesh.nvertices(), mesh.nlevels());

  auto pp_ = mesh.make_const_storage(pp, mesh.nvertices(), mesh.nlevels());
  auto sign_ = mesh.template make_const_storage<array<float_t, 6>>(
      sign, mesh.nvertices());
  auto vol_ = mesh.make_const_storage(vol, mesh.nvertices());
  auto s_ = mesh.template make_const_storage<tuple<float_t, float_t>>(
      s, mesh.nedges(), mesh.nlevels());

  auto v2e_tbl = mesh.v2e_table();
  auto v2e_conn =
      connectivity<generated::V2E_t>(v2e_tbl->get_const_target_ptr());

  auto e2v_tbl = mesh.e2v_table();
  auto e2v_conn =
      connectivity<generated::E2V_t>(e2v_tbl->get_const_target_ptr());
  auto vertex_domain = unstructured_domain({mesh.nvertices(), mesh.nlevels()},
                                           {}, v2e_conn, e2v_conn);

  generated::nabla_fencil(e2v_conn, v2e_conn)(fn_backend_t{}, mesh.nvertices(), mesh.nlevels(),
                               actual, pp_, s_, sign_, vol_);

  auto expected = make_expected(mesh);
  TypeParam::verify(expected, actual);
}

} // namespace
