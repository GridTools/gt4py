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

#include <gridtools/meta.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

#include "vertical_advection_defs.hpp"
#include "vertical_advection_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    // This is the definition of the special regions in the "vertical" direction
    using axis_t = axis<1, axis_config::offset_limit<3>>;
    using full_t = axis_t::full_interval;

    class u_forward_function {
        using utens_stage = in_accessor<0>;
        using wcon = in_accessor<1, extent<0, 1, 0, 0, 0, 1>>;
        using u_stage = in_accessor<2, extent<0, 0, 0, 0, -1, 1>>;
        using u_pos = in_accessor<3>;
        using utens = in_accessor<4>;
        using dtr_stage = in_accessor<5>;
        using ccol = inout_accessor<6, extent<0, 0, 0, 0, -1, 0>>;
        using dcol = inout_accessor<7, extent<0, 0, 0, 0, -1, 0>>;

        template <class Eval>
        GT_FUNCTION static auto compute_d(Eval &&eval) {
            return eval(dtr_stage()) * eval(u_pos()) + eval(utens()) + eval(utens_stage());
        }

      public:
        using param_list = make_param_list<utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, ccol, dcol>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, full_t::modify<1, -1>) {
            using float_t = std::decay_t<decltype(eval(ccol()))>;
            auto gav = -float_t(.25) * (eval(wcon(1, 0, 0)) + eval(wcon(0, 0, 0)));
            auto gcv = float_t(.25) * (eval(wcon(1, 0, 1)) + eval(wcon(0, 0, 1)));
            auto as = gav * float_t(BET_M);
            auto cs = gcv * float_t(BET_M);
            auto a = gav * float_t(BET_P);
            auto c = gcv * float_t(BET_P);
            auto b = eval(dtr_stage()) - a - c;
            auto correction =
                -as * (eval(u_stage(0, 0, -1)) - eval(u_stage())) - cs * (eval(u_stage(0, 0, 1)) - eval(u_stage()));
            auto d = compute_d(eval) + correction;
            auto divided = float_t(1) / (b - eval(ccol(0, 0, -1)) * a);

            eval(ccol()) = c * divided;
            eval(dcol()) = (d - eval(dcol(0, 0, -1)) * a) * divided;
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, full_t::last_level) {
            using float_t = std::decay_t<decltype(eval(ccol()))>;
            auto gav = -float_t(.25) * (eval(wcon(1, 0, 0)) + eval(wcon()));
            auto as = gav * float_t(BET_M);
            auto a = gav * float_t(BET_P);
            auto b = eval(dtr_stage()) - a;
            auto correction = -as * (eval(u_stage(0, 0, -1)) - eval(u_stage()));
            auto d = compute_d(eval) + correction;
            auto divided = float_t(1) / (b - eval(ccol(0, 0, -1)) * a);

            eval(dcol()) = (d - eval(dcol(0, 0, -1)) * a) * divided;
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, full_t::first_level) {
            using float_t = std::decay_t<decltype(eval(ccol()))>;
            auto gcv = float_t(.25) * (eval(wcon(1, 0, 1)) + eval(wcon(0, 0, 1)));
            auto cs = gcv * float_t(BET_M);
            auto c = gcv * float_t(BET_P);
            auto b = eval(dtr_stage()) - c;
            auto correction = -cs * (eval(u_stage(0, 0, 1)) - eval(u_stage()));
            auto d = compute_d(eval) + correction;
            auto divided = float_t(1) / b;

            eval(ccol()) = c * divided;
            eval(dcol()) = d * divided;
        }
    };

    class u_backward_function {
        using utens_stage = inout_accessor<0>;
        using u_pos = in_accessor<1>;
        using dtr_stage = in_accessor<2>;
        using ccol = in_accessor<3>;
        using dcol = in_accessor<4>;
        using data_col = inout_accessor<5, extent<0, 0, 0, 0, 0, 1>>;

      public:
        using param_list = make_param_list<utens_stage, u_pos, dtr_stage, ccol, dcol, data_col>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, full_t::modify<0, -1>) {
            auto data = eval(dcol()) - eval(ccol()) * eval(data_col(0, 0, 1));
            eval(utens_stage()) = eval(dtr_stage()) * (data - eval(u_pos()));
            eval(data_col()) = data;
        }

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval, full_t::last_level) {
            eval(utens_stage()) = eval(dtr_stage()) * (eval(dcol()) - eval(u_pos()));
            eval(data_col()) = eval(dcol());
        }
    };

    using modified_backend_t = meta::if_<meta::is_instantiation_of<gpu_backend::gpu, stencil_backend_t>,
        gpu_backend::gpu<integral_constant<int_t, 256>, integral_constant<int_t, 1>, integral_constant<int_t, 1>>,
        stencil_backend_t>;

    using env_t = vertical_test_environment<3, axis_t>;

    GT_REGRESSION_TEST(vertical_advection_dycore, env_t, modified_backend_t) {
        vertical_advection_repository repo{TypeParam::d(0), TypeParam::d(1), TypeParam::d(2)};
        auto utens_stage = TypeParam::make_storage(repo.utens_stage_in);
        auto comp = [&,
                        grid = TypeParam::make_grid(),
                        u_stage = TypeParam::make_storage(repo.u_stage),
                        wcon = TypeParam::make_storage(repo.wcon),
                        u_pos = TypeParam::make_storage(repo.u_pos),
                        utens = TypeParam::make_storage(repo.utens),
                        dtr_stage = global_parameter(typename TypeParam::float_t(repo.dtr_stage))] {
            const auto spec = [](auto utens_stage, auto u_stage, auto wcon, auto u_pos, auto utens, auto dtr_stage) {
                GT_DECLARE_TMP(typename TypeParam::float_t, ccol, dcol, data_col);
                return multi_pass(
                    execute_forward()
                        .k_cached(cache_io_policy::flush(), ccol, dcol)
                        .k_cached(cache_io_policy::fill(), u_stage)
                        .stage(u_forward_function(), utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, ccol, dcol),
                    execute_backward().k_cached(data_col).stage(
                        u_backward_function(), utens_stage, u_pos, dtr_stage, ccol, dcol, data_col));
            };
            run(spec, modified_backend_t(), grid, utens_stage, u_stage, wcon, u_pos, utens, dtr_stage);
        };
        comp();
        TypeParam::verify(repo.utens_stage_out, utens_stage);
        TypeParam::benchmark("vertical_advection_dycore", comp);
    }
} // namespace
