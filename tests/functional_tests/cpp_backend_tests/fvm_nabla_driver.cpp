#include <gtest/gtest.h>
#include <iostream>

#include "build/generated_fvm_nabla.hpp" // TODO
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/sid_shift_origin.hpp>

#include "simple_mesh.hpp"
#include <fn_select.hpp>
// #include <test_environment.hpp>

namespace {
using namespace gridtools;
using namespace fn;
using namespace literals;

TEST(unstructured, nabla) {
  using namespace simple_mesh;
  constexpr auto K = 3_c;
  constexpr auto n_v2e = 6_c;

  double pp[n_vertices][K];
  for (auto &ppp : pp)
    for (auto &p : ppp)
      p = rand() % 100;

  std::array<int, n_v2e> sign[n_vertices];
  for (auto &&ss : sign)
    for (auto &s : ss)
      s = rand() % 2 ? 1 : -1;

  double vol[n_vertices];
  for (auto &v : vol)
    v = rand() % 2 + 1;

  tuple<double, double> s[n_edges][K];
  for (auto &ss : s)
    for (auto &sss : ss)
      sss = {rand() % 100, rand() % 100};

  auto zavg = [&](int edge, int k) -> std::array<double, 2> {
    auto tmp = 0.;
    for (auto vertex : e2v[edge])
      tmp += pp[vertex][k];
    tmp /= 2;
    return {tmp * get<0>(s[edge][k]), tmp * get<1>(s[edge][k])};
  };

  auto expected = [&](int vertex, int k) {
    auto res = std::array{0., 0.};
    for (int i = 0; i != 2; ++i) {
      for (int j = 0; j != n_v2e; ++j) {
        auto edge = v2e[vertex][j];
        if (edge == -1)
          break;
        res[i] += zavg(edge, k)[i] * sign[vertex][j];
      }
      res[i] /= vol[vertex];
    }
    return res;
  };

  tuple<double, double> actual_zavg[n_edges][K] = {};

  auto e2v_conn = connectivity<generated::E2V_t>(&e2v[0]);
  auto edge_domain = unstructured_domain(n_edges, K, e2v_conn);

  generated::zavgS_fencil(edge_domain, actual_zavg, pp, s);

  for (int h = 0; h < n_edges; ++h)
    for (int v = 0; v < K; ++v)
      tuple_util::for_each(
          [](auto actual, auto expected) {
            EXPECT_DOUBLE_EQ(actual, expected);
          },
          actual_zavg[h][v], zavg(h, v));

  auto v2e_conn = connectivity<generated::V2E_t>(&v2e[0]);
  auto vertex_domain = unstructured_domain(n_vertices, K, e2v_conn, v2e_conn);

  tuple<double, double> actual[n_vertices][K] = {};
  generated::nabla_fencil(vertex_domain, actual, pp, s, sign, vol);

  for (int h = 0; h < n_vertices; ++h)
    for (int v = 0; v < K; ++v)
      tuple_util::for_each(
          [](auto actual, auto expected) {
            EXPECT_DOUBLE_EQ(actual, expected);
          },
          actual[h][v], expected(h, v));
}
} // namespace
