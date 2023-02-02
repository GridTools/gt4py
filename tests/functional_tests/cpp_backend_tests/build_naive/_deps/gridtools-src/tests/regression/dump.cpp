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
#include <gridtools/stencil/dump.hpp>

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
        eval(out()) = float_t{4} * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

struct flx_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<0, 1, 0, 0>>;
    using lap = in_accessor<2, extent<0, 1, 0, 0>>;

    using param_list = make_param_list<out, in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto res = eval(lap(1, 0)) - eval(lap(0, 0));
        eval(out()) = res * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0 : res;
    }
};

struct fly_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<0, 0, 0, 1>>;
    using lap = in_accessor<2, extent<0, 0, 0, 1>>;

    using param_list = make_param_list<out, in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto res = eval(lap(0, 1)) - eval(lap(0, 0));
        eval(out()) = res * (eval(in(0, 1)) - eval(in(0, 0))) > 0 ? 0 : res;
    }
};

struct out_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1>;
    using flx = in_accessor<2, extent<-1, 0, 0, 0>>;
    using fly = in_accessor<3, extent<0, 0, -1, 0>>;
    using coeff = in_accessor<4>;

    using param_list = make_param_list<out, in, flx, fly, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0)) + eval(fly()) - eval(fly(0, -1)));
    }
};

auto const horizontal_diffusion = [](auto in, auto coeff, auto out) {
    GT_DECLARE_TMP(double, lap, flx, fly);
    return execute_parallel()
        .ij_cached(lap, flx, fly)
        .stage(lap_function(), lap, in)
        .stage(flx_function(), flx, in, lap)
        .stage(fly_function(), fly, in, lap)
        .stage(out_function(), out, in, flx, fly, coeff);
};

TEST(dump, horizontal_diffusion) {
    double fake[5][5][1];
    halo_descriptor hd(2, 2, 2, 2, 5);
    run(horizontal_diffusion, dump{std::cout << std::setw(1)}, make_grid(hd, hd, 1), fake, fake, fake);
}
