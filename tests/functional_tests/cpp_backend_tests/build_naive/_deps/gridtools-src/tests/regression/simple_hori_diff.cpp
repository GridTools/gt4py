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

#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

#include "horizontal_diffusion_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct wlap_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 1, -1, 1>>;
        using crlato = in_accessor<2>;
        using crlatu = in_accessor<3>;

        using param_list = make_param_list<out, in, crlato, crlatu>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            using float_t = std::decay_t<decltype(eval(out()))>;
            eval(out()) = eval(in(1, 0)) + eval(in(-1, 0)) - float_t{2} * eval(in()) +
                          eval(crlato()) * (eval(in(0, 1)) - eval(in())) +
                          eval(crlatu()) * (eval(in(0, -1)) - eval(in()));
        }
    };

    struct divflux_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1>;
        using lap = in_accessor<2, extent<-1, 1, -1, 1>>;
        using crlato = in_accessor<3>;
        using coeff = in_accessor<4>;

        using param_list = make_param_list<out, in, lap, crlato, coeff>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            auto fluxx = eval(lap(1, 0)) - eval(lap());
            auto fluxx_m = eval(lap()) - eval(lap(-1, 0));

            auto fluxy = eval(crlato()) * (eval(lap(0, 1)) - eval(lap()));
            auto fluxy_m = eval(crlato()) * (eval(lap()) - eval(lap(0, -1)));

            eval(out()) = eval(in()) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * eval(coeff());
        }
    };

    GT_REGRESSION_TEST(simple_hori_diff, test_environment<2>, stencil_backend_t) {
        const auto j_builder = TypeParam::builder().template selector<0, 1, 0>();
        horizontal_diffusion_repository repo(TypeParam::d(0), TypeParam::d(1), TypeParam::d(2));
        auto out = TypeParam::make_storage();
        auto comp = [grid = TypeParam::make_grid(),
                        coeff = TypeParam::make_storage(repo.coeff),
                        in = TypeParam::make_storage(repo.in),
                        &out,
                        crlato = j_builder.initializer(repo.crlato)(),
                        crlatu = j_builder.initializer(repo.crlatu)()] {
            const auto spec = [](auto coeff, auto in, auto out, auto crlato, auto crlatu) {
                GT_DECLARE_TMP(typename TypeParam::float_t, lap);
                return execute_parallel()
                    .ij_cached(lap)
                    .stage(wlap_function(), lap, in, crlato, crlatu)
                    .stage(divflux_function(), out, in, lap, crlato, coeff);
            };
            run(spec, stencil_backend_t(), grid, coeff, in, out, crlato, crlatu);
        };
        comp();
        TypeParam::verify(repo.out_simple, out);
        TypeParam::benchmark("simple_hori_diff", comp);
    }
} // namespace
