
#include <gridtools/common/array.hpp>
#include <gridtools/common/gt_math.hpp>
#include <gridtools/common/gt_math.hpp>
#include <gridtools/usid/dim.hpp>
#include <gridtools/usid/helpers.hpp>
#include <gridtools/usid/naive_helpers.hpp>

namespace nabla_impl_ {
using namespace gridtools;
using namespace gridtools::usid;
using namespace gridtools::usid::naive;
struct v2e_tag : connectivity<7, true> {};
struct e2v_tag : connectivity<2, false> {};
struct S_MXX_tag;
struct pp_tag;
struct pnabla_MXX_tag;
struct vol_tag;
struct zavg_tmp_tag;
struct sign_tag;
struct zavgS_MYY_tag;
struct zavgS_MXX_tag;
struct pnabla_MYY_tag;
struct S_MYY_tag;

struct nabla_edge_1 {
  GT_FUNCTION auto operator()() const {
    return [](auto &&primary_edge_ptrs, auto &&primary_edge_strides,
              auto &&vertices) {
      field<zavg_tmp_tag>(primary_edge_ptrs) = (double)0.0;
      foreach_neighbor<e2v_tag>(
          [&](auto &&prim, auto &&sec) {
            field<zavg_tmp_tag>(prim) =
                (field<zavg_tmp_tag>(prim) + field<pp_tag>(sec));
          },
          primary_edge_ptrs, primary_edge_strides, vertices);
      field<zavg_tmp_tag>(primary_edge_ptrs) =
          ((double)0.5 * field<zavg_tmp_tag>(primary_edge_ptrs));
      field<zavgS_MXX_tag>(primary_edge_ptrs) =
          (field<S_MXX_tag>(primary_edge_ptrs) *
           field<zavg_tmp_tag>(primary_edge_ptrs));
      field<zavgS_MYY_tag>(primary_edge_ptrs) =
          (field<S_MYY_tag>(primary_edge_ptrs) *
           field<zavg_tmp_tag>(primary_edge_ptrs));
    };
  }
};

struct nabla_vertex_2 {
  GT_FUNCTION auto operator()() const {
    return [](auto &&vertex_prim_ptrs, auto &&vertex_prim_strides,
              auto &&edge_neighbors) {
      field<pnabla_MXX_tag>(vertex_prim_ptrs) = (double)0.0;
      foreach_neighbor<v2e_tag>(
          [&](auto &&p, auto &&n) {
            field<pnabla_MXX_tag>(p) =
                (field<pnabla_MXX_tag>(p) +
                 (field<zavgS_MXX_tag>(n) * field<sign_tag>(p)));
          },
          vertex_prim_ptrs, vertex_prim_strides, edge_neighbors);
      field<pnabla_MYY_tag>(vertex_prim_ptrs) = (double)0.0;
      foreach_neighbor<v2e_tag>(
          [&](auto &&p, auto &&n) {
            field<pnabla_MYY_tag>(p) =
                (field<pnabla_MYY_tag>(p) +
                 (field<zavgS_MYY_tag>(n) * field<sign_tag>(p)));
          },
          vertex_prim_ptrs, vertex_prim_strides, edge_neighbors);
    };
  }
};

struct nabla_vertex_4 {
  GT_FUNCTION auto operator()() const {
    return [](auto &&primary_ptrs, auto &&primary_strides) {
      field<pnabla_MXX_tag>(primary_ptrs) =
          (field<pnabla_MXX_tag>(primary_ptrs) / field<vol_tag>(primary_ptrs));
      field<pnabla_MYY_tag>(primary_ptrs) =
          (field<pnabla_MYY_tag>(primary_ptrs) / field<vol_tag>(primary_ptrs));
    };
  }
};

auto nabla = [](domain d, auto &&v2e, auto &&e2v) {
  // TODO assert connectivities are sid
  return [d = std::move(d),
          v2e = sid::rename_dimensions<dim::n, v2e_tag>(
              std::forward<decltype(v2e)>(v2e)(traits_t())),
          e2v = sid::rename_dimensions<dim::n, e2v_tag>(
              std::forward<decltype(e2v)>(e2v)(traits_t()))](
             auto &&S_MXX, auto &&S_MYY, auto &&pp, auto &&pnabla_MXX,
             auto &&pnabla_MYY, auto &&vol, auto &&sign) {
    // TODO assert field params are sids
    auto alloc = make_allocator();

    auto zavgS_MXX = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
    auto zavgS_MYY = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
    auto zavg_tmp = make_simple_tmp_storage<double>(d.edge, d.k, alloc);

    call_kernel<nabla_edge_1>(
        d.edge, d.k,
        sid::composite::make<e2v_tag, zavg_tmp_tag, S_MXX_tag, S_MYY_tag,
                             zavgS_MXX_tag, zavgS_MYY_tag>(
            e2v, zavg_tmp, S_MXX, S_MYY, zavgS_MXX, zavgS_MYY),
        sid::composite::make<pp_tag>(pp));

    call_kernel<nabla_vertex_2>(
        d.vertex, d.k,
        sid::composite::make<v2e_tag, pnabla_MXX_tag, pnabla_MYY_tag, sign_tag>(
            v2e, pnabla_MXX, pnabla_MYY,
            sid::rename_dimensions<dim::s, v2e_tag>(sign)),
        sid::composite::make<zavgS_MXX_tag, zavgS_MYY_tag>(zavgS_MXX,
                                                           zavgS_MYY));

    call_kernel<nabla_vertex_4>(
        d.vertex, d.k,
        sid::composite::make<pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>(
            pnabla_MXX, pnabla_MYY, vol));
  };
};
} // namespace nabla_impl_

using nabla_impl_::nabla;
