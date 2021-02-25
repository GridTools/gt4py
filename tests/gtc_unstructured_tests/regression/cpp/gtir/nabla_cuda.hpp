
#include <gridtools/common/array.hpp>
#include <gridtools/common/gt_math.hpp>
#include <gridtools/common/gt_math.hpp>
#include <gridtools/usid/cuda_helpers.hpp>
#include <gridtools/usid/dim.hpp>
#include <gridtools/usid/helpers.hpp>

namespace nabla_impl_ {
using namespace gridtools;
using namespace gridtools::usid;
using namespace gridtools::usid::cuda;
struct v2e_tag : connectivity<7, true> {};
struct e2v_tag : connectivity<2, false> {};
struct S_MXX_tag;
struct zavg_tmp_tag;
struct S_MYY_tag;
struct vol_tag;
struct pnabla_MXX_tag;
struct sign_tag;
struct zavgS_MXX_tag;
struct zavgS_MYY_tag;
struct pp_tag;
struct pnabla_MYY_tag;

struct kernel_HorizontalLoop_244 {
  GT_FUNCTION auto operator()() const {
    return [](auto &&e_ptrs, auto &&e_strides, auto &&e2v_sid) {
      double localNeighborReduce_26 = (double)0.0;
      localNeighborReduce_26 = (double)0;
      foreach_neighbor<e2v_tag>(
          [&](auto &&p, auto &&n) {
            localNeighborReduce_26 =
                (localNeighborReduce_26 + field<pp_tag>(n));
          },
          e_ptrs, e_strides, e2v_sid);
      field<zavg_tmp_tag>(e_ptrs) = ((double)0.5 * localNeighborReduce_26);
      field<zavgS_MXX_tag>(e_ptrs) =
          (field<zavg_tmp_tag>(e_ptrs) * field<S_MXX_tag>(e_ptrs));
      field<zavgS_MYY_tag>(e_ptrs) =
          (field<zavg_tmp_tag>(e_ptrs) * field<S_MYY_tag>(e_ptrs));
    };
  }
};

struct kernel_HorizontalLoop_246 {
  GT_FUNCTION auto operator()() const {
    return [](auto &&v_ptrs, auto &&v_strides, auto &&v2e_sid) {
      double localNeighborReduce_65 = (double)0.0;
      double localNeighborReduce_77 = (double)0.0;
      localNeighborReduce_65 = (double)0;
      foreach_neighbor<v2e_tag>(
          [&](auto &&p, auto &&n) {
            localNeighborReduce_65 =
                (localNeighborReduce_65 +
                 (field<zavgS_MXX_tag>(n) * field<sign_tag>(p)));
          },
          v_ptrs, v_strides, v2e_sid);
      field<pnabla_MXX_tag>(v_ptrs) = localNeighborReduce_65;
      localNeighborReduce_77 = (double)0;
      foreach_neighbor<v2e_tag>(
          [&](auto &&p, auto &&n) {
            localNeighborReduce_77 =
                (localNeighborReduce_77 +
                 (field<zavgS_MYY_tag>(n) * field<sign_tag>(p)));
          },
          v_ptrs, v_strides, v2e_sid);
      field<pnabla_MYY_tag>(v_ptrs) = localNeighborReduce_77;
    };
  }
};

struct kernel_HorizontalLoop_248 {
  GT_FUNCTION auto operator()() const {
    return [](auto &&v_ptrs, auto &&v_strides) {
      field<pnabla_MXX_tag>(v_ptrs) =
          (field<pnabla_MXX_tag>(v_ptrs) / field<vol_tag>(v_ptrs));
      field<pnabla_MYY_tag>(v_ptrs) =
          (field<pnabla_MYY_tag>(v_ptrs) / field<vol_tag>(v_ptrs));
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

    auto zavg_tmp = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
    auto zavgS_MXX = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
    auto zavgS_MYY = make_simple_tmp_storage<double>(d.edge, d.k, alloc);

    call_kernel<kernel_HorizontalLoop_244>(
        d.edge, d.k,
        sid::composite::make<S_MXX_tag, zavgS_MYY_tag, zavgS_MXX_tag,
                             zavg_tmp_tag, S_MYY_tag, e2v_tag>(
            S_MXX, zavgS_MYY, zavgS_MXX, zavg_tmp, S_MYY, e2v),
        sid::composite::make<pp_tag>(pp));

    call_kernel<kernel_HorizontalLoop_246>(
        d.vertex, d.k,
        sid::composite::make<pnabla_MYY_tag, pnabla_MXX_tag, v2e_tag, sign_tag>(
            pnabla_MYY, pnabla_MXX, v2e,
            sid::rename_dimensions<dim::s, v2e_tag>(sign)),
        sid::composite::make<zavgS_MYY_tag, zavgS_MXX_tag>(zavgS_MYY,
                                                           zavgS_MXX));

    call_kernel<kernel_HorizontalLoop_248>(
        d.vertex, d.k,
        sid::composite::make<vol_tag, pnabla_MXX_tag, pnabla_MYY_tag>(
            vol, pnabla_MXX, pnabla_MYY));
  };
};
} // namespace nabla_impl_

using nabla_impl_::nabla;
