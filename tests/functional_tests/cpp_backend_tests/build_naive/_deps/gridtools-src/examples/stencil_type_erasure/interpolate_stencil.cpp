/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "interpolate_stencil.hpp"

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GT_CUDACC
#include <gridtools/stencil/gpu.hpp>
using stencil_backend_t = gridtools::stencil::gpu<>;
#else
#include <gridtools/stencil/cpu_ifirst.hpp>
using stencil_backend_t = gridtools::stencil::cpu_ifirst<>;
#endif

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct interpolate_stage {
        using in1 = in_accessor<0>;
        using in2 = in_accessor<1>;
        using weight = in_accessor<2>;
        using out = inout_accessor<3>;

        using param_list = make_param_list<in1, in2, weight, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            using namespace expressions;
            eval(out()) = eval(weight() * in1() + (1. - weight()) * in2());
        }
    };
} // namespace

// `run_single_stage` should never be called in a header, because the compilation overhead is very significant
std::function<void(inputs, outputs)> make_interpolate_stencil(grid_t grid, double weight) {
    return [grid = std::move(grid), weight](inputs in, outputs out) {
        run_single_stage(
            interpolate_stage(), stencil_backend_t(), grid, in.in1, in.in2, global_parameter(weight), out.out);
    };
}
