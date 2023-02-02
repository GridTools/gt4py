/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/fn/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

#include "../vertical_advection_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace cartesian;
    using namespace literals;
    using stencil::global_parameter;

    struct u_forward_scan : fwd {
        static GT_FUNCTION constexpr auto prologue() {
            return tuple(scan_pass(
                [](auto /*acc*/,
                    auto const &utens_stage,
                    auto const &utens,
                    auto const &u_stage,
                    auto const &u_pos,
                    auto const &wcon,
                    auto const &dtr_stage) {
                    constexpr auto i = dim::i();
                    constexpr auto k = dim::k();
                    using float_t = std::decay_t<decltype(deref(wcon))>;
                    auto gcv = float_t(0.25) * (deref(shift(wcon, i, 1, k, 1)) + deref(shift(wcon, k, 1)));
                    auto cs = gcv * float_t(BET_M);
                    auto c = gcv * float_t(BET_P);
                    auto b = deref(dtr_stage) - c;
                    auto correction = -cs * (deref(shift(u_stage, k, 1)) - deref(u_stage));
                    auto d = deref(dtr_stage) * deref(u_pos) + deref(utens) + deref(utens_stage) + correction;
                    auto divided = float_t(1) / b;
                    return make_tuple(c * divided, d * divided);
                },
                host_device::identity()));
        }

        static GT_FUNCTION constexpr auto body() {
            return scan_pass(
                [](auto acc,
                    auto const &utens_stage,
                    auto const &utens,
                    auto const &u_stage,
                    auto const &u_pos,
                    auto const &wcon,
                    auto const &dtr_stage) {
                    constexpr auto i = dim::i();
                    constexpr auto k = dim::k();
                    using float_t = std::decay_t<decltype(deref(wcon))>;
                    auto gav = -float_t(0.25) * (deref(shift(wcon, i, 1)) + deref(wcon));
                    auto gcv = float_t(0.25) * (deref(shift(wcon, i, 1, k, 1)) + deref(shift(wcon, k, 1)));
                    auto as = gav * float_t(BET_M);
                    auto cs = gcv * float_t(BET_M);
                    auto a = gav * float_t(BET_P);
                    auto c = gcv * float_t(BET_P);
                    auto b = deref(dtr_stage) - a - c;
                    auto correction = -as * (deref(shift(u_stage, k, -1)) - deref(u_stage)) -
                                      cs * (deref(shift(u_stage, k, -1)) - deref(u_stage));
                    auto d = deref(dtr_stage) * deref(u_pos) + deref(utens) + deref(utens_stage) + correction;
                    auto [cp, dp] = acc;
                    auto divided = float_t(1) / (b - cp * a);
                    return make_tuple(c * divided, (d - dp * a) * divided);
                },
                host_device::identity());
        }

        static GT_FUNCTION constexpr auto epilogue() {
            return tuple(scan_pass(
                [](auto acc,
                    auto const &utens_stage,
                    auto const &utens,
                    auto const &u_stage,
                    auto const &u_pos,
                    auto const &wcon,
                    auto const &dtr_stage) {
                    constexpr auto i = dim::i();
                    constexpr auto k = dim::k();
                    using float_t = std::decay_t<decltype(deref(wcon))>;
                    auto gav = -float_t(0.25) * (deref(shift(wcon, i, 1)) + deref(wcon));
                    auto as = gav * float_t(BET_M);
                    auto a = gav * float_t(BET_P);
                    auto b = deref(dtr_stage) - a;
                    auto correction = -as * (deref(shift(u_stage, k, -1)) - deref(u_stage));
                    auto d = deref(dtr_stage) * deref(u_pos) + deref(utens) + deref(utens_stage) + correction;
                    auto [cp, dp] = acc;
                    auto divided = float_t(1) / (b - cp * a);
                    return make_tuple(float_t(0), (d - dp * a) * divided);
                },
                host_device::identity()));
        }
    };

    struct u_backward_scan : bwd {
        static GT_FUNCTION constexpr auto prologue() {
            return tuple(scan_pass(
                [](auto /*acc*/, auto const &cd, auto const &u_pos, auto const &dtr_stage) {
                    auto d = tuple_get(1_c, deref(cd));
                    return make_tuple(deref(dtr_stage) * (d - deref(u_pos)), d);
                },
                [](auto const &acc) { return tuple_get(0_c, acc); }));
        }

        static GT_FUNCTION constexpr auto body() {
            return scan_pass(
                [](auto acc, auto const &cd, auto const &u_pos, auto const &dtr_stage) {
                    auto [c, d] = deref(cd);
                    auto data = d - c * tuple_get(1_c, acc);
                    return tuple(deref(dtr_stage) * (data - deref(u_pos)), data);
                },
                [](auto const &acc) { return tuple_get(0_c, acc); });
        }
    };

    constexpr inline auto vadv_solver = [](auto &&executor,
                                            auto &cd,
                                            auto &utens_stage,
                                            auto const &utens,
                                            auto const &u_stage,
                                            auto const &u_pos,
                                            auto const &wcon,
                                            auto const &dtr_stage) {
        using float_t = sid::element_type<decltype(utens_stage)>;
        executor()
            .arg(cd)
            .arg(utens_stage)
            .arg(utens)
            .arg(u_stage)
            .arg(u_pos)
            .arg(wcon)
            .arg(dtr_stage)
            .assign(0_c, u_forward_scan(), tuple<float_t, float_t>(0, 0), 1_c, 2_c, 3_c, 4_c, 5_c, 6_c)
            .assign(1_c, u_backward_scan(), tuple<float_t, float_t>(0, 0), 0_c, 4_c, 6_c)
            .execute();
    };

    GT_REGRESSION_TEST(fn_cartesian_vertical_advection, vertical_test_environment<3>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;
        vertical_advection_repository repo{TypeParam::d(0), TypeParam::d(1), TypeParam::d(2)};

        auto fencil = [](int i,
                          int j,
                          int k,
                          auto &utens_stage,
                          auto const &utens,
                          auto const &u_stage,
                          auto const &u_pos,
                          auto const &wcon,
                          auto const &dtr_stage) {
            using sizes_t = hymap::keys<dim::i, dim::j, dim::k>::values<int, int, int>;
            auto be = fn_backend_t();
            auto domain = cartesian_domain(sizes_t(i - 6, j - 6, k), sizes_t(3, 3, 0));
            auto backend = make_backend(be, domain);
            auto alloc = tmp_allocator(be);
            auto cd = allocate_global_tmp<tuple<float_t, float_t>>(alloc, sizes_t(i, j, k));
            vadv_solver(backend.vertical_executor(), cd, utens_stage, utens, u_stage, u_pos, wcon, dtr_stage);
        };

        auto utens_stage = TypeParam::make_storage(repo.utens_stage_in);
        auto comp = [&,
                        utens = TypeParam::make_storage(repo.utens),
                        u_stage = TypeParam::make_storage(repo.u_stage),
                        u_pos = TypeParam::make_storage(repo.u_pos),
                        wcon = TypeParam::make_storage(repo.wcon),
                        dtr_stage = stencil::global_parameter(float_t(repo.dtr_stage))] {
            fencil(
                TypeParam::d(0), TypeParam::d(1), TypeParam::d(2), utens_stage, utens, u_stage, u_pos, wcon, dtr_stage);
        };
        comp();
        TypeParam::verify(repo.utens_stage_out, utens_stage);
        TypeParam::benchmark("fn_cartesian_vertical_advection", comp);
    }
} // namespace
