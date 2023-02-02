/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

#include "horizontal_diffusion_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct lap_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 1, -1, 1>>;

        using param_list = make_param_list<out, in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            using float_t = std::decay_t<decltype(eval(out()))>;
            eval(out()) =
                float_t{4} * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
        }
    };

    struct flx_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 2, -1, 1>>;

        using param_list = make_param_list<out, in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            auto lap_hi = call<lap_function>::with(eval, in(1, 0));
            auto lap_lo = call<lap_function>::with(eval, in(0, 0));
            auto flx = lap_hi - lap_lo;
            eval(out()) = flx * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0 : flx;
        }
    };

    struct fly_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 1, -1, 2>>;

        using param_list = make_param_list<out, in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            auto lap_hi = call<lap_function>::with(eval, in(0, 1));
            auto lap_lo = call<lap_function>::with(eval, in(0, 0));
            auto fly = lap_hi - lap_lo;
            eval(out()) = fly * (eval(in(0, 1)) - eval(in(0, 0))) > 0 ? 0 : fly;
        }
    };

    struct out_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-2, 2, -2, 2>>;
        using coeff = in_accessor<2>;

        using param_list = make_param_list<out, in, coeff>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            auto flx_hi = call<flx_function>::with(eval, in(0, 0));
            auto flx_lo = call<flx_function>::with(eval, in(-1, 0));

            auto fly_hi = call<fly_function>::with(eval, in(0, 0));
            auto fly_lo = call<fly_function>::with(eval, in(0, -1));

            eval(out()) = eval(in()) - eval(coeff()) * (flx_hi - flx_lo + fly_hi - fly_lo);
        }
    };

    GT_REGRESSION_TEST(horizontal_diffusion_fused, test_environment<2>, stencil_backend_t) {
        auto out = TypeParam::make_storage();

        horizontal_diffusion_repository repo(TypeParam::d(0), TypeParam::d(1), TypeParam::d(2));

        auto comp = [&,
                        grid = TypeParam::make_grid(),
                        in = TypeParam::make_storage(repo.in),
                        coeff = TypeParam::make_storage(repo.coeff)] {
            run_single_stage(out_function(), stencil_backend_t(), grid, out, in, coeff);
        };

        comp();
        TypeParam::verify(repo.out, out);
        TypeParam::benchmark("horizontal_diffusion_fused", comp);
    }
} // namespace
