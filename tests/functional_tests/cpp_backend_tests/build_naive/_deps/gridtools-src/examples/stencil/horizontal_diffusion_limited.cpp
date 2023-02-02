/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
   @file This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO since
   it implements flux-limiting
*/

#include <cstdlib>
#include <iostream>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GT_CUDACC
#include <gridtools/stencil/gpu.hpp>
#include <gridtools/storage/gpu.hpp>
using stencil_backend_t = gridtools::stencil::gpu<>;
using storage_traits_t = gridtools::storage::gpu;
#else
#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
using stencil_backend_t = gridtools::stencil::cpu_ifirst<>;
using storage_traits_t = gridtools::storage::cpu_ifirst;
#endif

namespace gt = gridtools;
namespace st = gt::stencil;

// These are the stencil operators that compose the multistage stencil in this test
struct lap_function {
    using out = st::cartesian::inout_accessor<0>;
    using in = st::cartesian::in_accessor<1, st::extent<-1, 1, -1, 1>>;

    using param_list = st::make_param_list<out, in>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = 4. * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

struct flx_function {
    using out = st::cartesian::inout_accessor<0>;
    using in = st::cartesian::in_accessor<1, st::extent<0, 1, 0, 0>>;
    using lap = st::cartesian::in_accessor<2, st::extent<0, 1, 0, 0>>;

    using param_list = st::make_param_list<out, in, lap>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        auto res = eval(lap(1, 0)) - eval(lap(0, 0));
        eval(out()) = res * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0 : res;
    }
};

struct fly_function {
    using out = st::cartesian::inout_accessor<0>;
    using in = st::cartesian::in_accessor<1, st::extent<0, 0, 0, 1>>;
    using lap = st::cartesian::in_accessor<2, st::extent<0, 0, 0, 1>>;

    using param_list = st::make_param_list<out, in, lap>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        auto res = eval(lap(0, 1)) - eval(lap(0, 0));
        eval(out()) = res * (eval(in(0, 1)) - eval(in(0, 0))) > 0 ? 0 : res;
    }
};

struct out_function {
    using out = st::cartesian::inout_accessor<0>;
    using in = st::cartesian::in_accessor<1>;
    using flx = st::cartesian::in_accessor<2, st::extent<-1, 0, 0, 0>>;
    using fly = st::cartesian::in_accessor<3, st::extent<0, 0, -1, 0>>;
    using coeff = st::cartesian::in_accessor<4>;

    using param_list = st::make_param_list<out, in, flx, fly, coeff>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0)) + eval(fly()) - eval(fly(0, -1)));
    }
};

int main(int argc, char **argv) {
    constexpr unsigned halo = 2;

    unsigned d1, d2, d3;
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    // Add dimensions and halos to the storage builder
    auto storage_builder = gt::storage::builder<storage_traits_t>.dimensions(d1, d2, d3).halos(halo, halo, 0);

    // Definition of the actual data fields that are used for input/output
    auto in = storage_builder.type<double const>().value(42).build();
    auto coeff = storage_builder.type<double const>().value(42).build();
    auto out = storage_builder.type<double>().build();

    // Here we specify the stencil composition of the computation.
    auto spec = [](auto coeff, auto in, auto out) {
        // temporary data (the library will take care of that and it is not observable by the user)
        GT_DECLARE_TMP(double, lap, flx, fly);
        return st::execute_parallel()
            .ij_cached(lap, flx, fly)
            .stage(lap_function(), lap, in)
            .stage(flx_function(), flx, in, lap)
            .stage(fly_function(), fly, in, lap)
            .stage(out_function(), out, in, flx, fly, coeff);
    };

    // Now we describe the iteration space. The first two dimensions are described with a tuple of values (minus, plus,
    // begin, end, length) begin and end, for each dimension represent the space where the output data will be located
    // in the data_stores, while minus and plus indicate the number of halo points in the indices before begin and after
    // end, respectively. The length, is not needed, and will be removed in future versions, but we keep it for now
    // since the data structure used is the same used in the communication library and there the length is used.
    gt::halo_descriptor di{halo, halo, halo, d1 - halo - 1, d1};
    gt::halo_descriptor dj{halo, halo, halo, d2 - halo - 1, d2};

    // The grid represent the iteration space. The third dimension is indicated here as a size and the iteration space
    // is deduced by the fact that there is not an axis definition. More complex third dimensions are possible but not
    // described in this example.
    auto grid = st::make_grid(di, dj, d3);

    // Here we perform the computation, specifying the backend, the grid (iteration space), binding spec arguments to
    // the data fields
    st::run(spec, stencil_backend_t(), grid, coeff, in, out);
}
