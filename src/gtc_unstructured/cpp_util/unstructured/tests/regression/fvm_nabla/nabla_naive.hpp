#pragma once
#include "gridtools/usid/dim.hpp"
#include "gridtools/usid/helpers.hpp"
#include <gridtools/usid/naive_helpers.hpp>
#include <iostream>
namespace gridtools::usid::naive::nabla_impl_ {
    struct v2e_tag : connectivity<7, true> {};
    struct e2v_tag : connectivity<2, false> {};
    struct S_MXX_tag;
    struct S_MYY_tag;
    struct zavgS_MXX_tag;
    struct zavgS_MYY_tag;
    struct pnabla_MXX_tag;
    struct pnabla_MYY_tag;
    struct vol_tag;
    struct sign_tag : sparse_field<v2e_tag> {};
    struct pp_tag;
    struct kernel_0 {
        GT_FUNCTION auto operator()() const {
            return [](auto &&ptr, auto &&strides, auto &&neighbors) {
                double acc = 0.0;
                foreach_neighbor<e2v_tag>([&](auto &&, auto &&n) { acc += field<pp_tag>(n); }, ptr, strides, neighbors);
                auto zavg = 0.5 * acc;
                field<zavgS_MXX_tag>(ptr) = field<S_MXX_tag>(ptr) * zavg;
                field<zavgS_MYY_tag>(ptr) = field<S_MYY_tag>(ptr) * zavg;
            };
        }
    };
    struct kernel_1 {
        GT_FUNCTION auto operator()() const {
            return [](auto &&ptr, auto &&strides, auto &&neighbors) {
                field<pnabla_MXX_tag>(ptr) = 0.0;
                field<pnabla_MYY_tag>(ptr) = 0.0;
                foreach_neighbor<v2e_tag>(
                    [](auto &&p, auto &&n) {
                        field<pnabla_MXX_tag>(p) += field<zavgS_MXX_tag>(n) * field<sign_tag>(p);
                        field<pnabla_MYY_tag>(p) += field<zavgS_MYY_tag>(n) * field<sign_tag>(p);
                    },
                    ptr,
                    strides,
                    neighbors);
            };
        }
    };

    struct kernel_2 {
        GT_FUNCTION auto operator()() const {
            return [](auto &&ptr, auto &&strides) {
                field<pnabla_MXX_tag>(ptr) = field<pnabla_MXX_tag>(ptr) / field<vol_tag>(ptr);
                field<pnabla_MYY_tag>(ptr) = field<pnabla_MYY_tag>(ptr) / field<vol_tag>(ptr);
            };
        }
    };

    inline constexpr auto nabla = [](domain d, auto &&v2e, auto &&e2v) {
        static_assert(is_sid<decltype(v2e(traits_t()))>());
        static_assert(is_sid<decltype(e2v(traits_t()))>());
        return
            [d = std::move(d),
                v2e = sid::rename_dimensions<dim::n, v2e_tag>(std::forward<decltype(v2e)>(v2e)(traits_t())),
                e2v = sid::rename_dimensions<dim::n, e2v_tag>(std::forward<decltype(e2v)>(e2v)(traits_t()))](
                auto &&S_MXX, auto &&S_MYY, auto &&pp, auto &&pnabla_MXX, auto &&pnabla_MYY, auto &&vol, auto &&sign) {
                static_assert(is_sid<decltype(S_MXX)>());
                static_assert(is_sid<decltype(S_MYY)>());
                static_assert(is_sid<decltype(pp)>());
                static_assert(is_sid<decltype(pnabla_MXX)>());
                static_assert(is_sid<decltype(pnabla_MYY)>());
                static_assert(is_sid<decltype(vol)>());
                static_assert(is_sid<decltype(sign)>());
                auto alloc = make_allocator();
                auto zavgS_MXX = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
                auto zavgS_MYY = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
                call_kernel<kernel_0>(d.edge,
                    sid::composite::make<e2v_tag, S_MXX_tag, S_MYY_tag, zavgS_MXX_tag, zavgS_MYY_tag>(
                        e2v, S_MXX, S_MYY, zavgS_MXX, zavgS_MYY),
                    sid::composite::make<pp_tag>(pp));
                call_kernel<kernel_1>(d.vertex,
                    sid::composite::make<v2e_tag, pnabla_MXX_tag, pnabla_MYY_tag, sign_tag>(
                        v2e, pnabla_MXX, pnabla_MYY, sid::rename_dimensions<dim::s, v2e_tag>(sign)),
                    sid::composite::make<zavgS_MXX_tag, zavgS_MYY_tag>(zavgS_MXX, zavgS_MYY));
                // TODO pole edge computation
                call_kernel<kernel_2>(d.vertex,
                    sid::composite::make<v2e_tag, pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>(
                        v2e, pnabla_MXX, pnabla_MYY, vol));
            };
    };

} // namespace gridtools::usid::naive::nabla_impl_
using gridtools::usid::naive::nabla_impl_::nabla;
