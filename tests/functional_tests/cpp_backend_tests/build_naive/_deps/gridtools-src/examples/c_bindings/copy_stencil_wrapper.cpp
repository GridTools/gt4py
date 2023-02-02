/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// In this example, we demonstrate how the cpp_bindgen library can be used to export functions to C and Fortran. We are
// going to export the functions required to run a simple copy stencil (see also the commented example in
// examples/stencil/copy_stencil.cpp)

#include <cassert>
#include <functional>

#include <cpp_bindgen/export.hpp>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/storage/adapter/fortran_array_adapter.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GT_CUDACC
#include <gridtools/common/cuda_runtime.hpp>
#include <gridtools/common/cuda_util.hpp>
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

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct copy_functor {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;
        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(in());
        }
    };

    using data_store_t = decltype(storage::builder<storage_traits_t>.type<float>().dimensions(0, 0, 0).build());

    auto make_data_store_impl(int x, int y, int z) {
        return storage::builder<storage_traits_t>.type<float>().dimensions(x, y, z).build();
    }
    BINDGEN_EXPORT_BINDING_3(make_data_store, make_data_store_impl);

    void run_copy_stencil_impl(data_store_t in, data_store_t out) {
        assert(in->lengths() == out->lengths());
        auto &&lengths = out->lengths();
        auto grid = make_grid(lengths[0], lengths[1], lengths[2]);
        run_single_stage(copy_functor(), stencil_backend_t(), grid, in, out);
#ifdef GT_CUDACC
        GT_CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }
    BINDGEN_EXPORT_BINDING_2(run_copy_stencil, run_copy_stencil_impl);

    template <class D>
    void transform_f_to_c_impl(D c, fortran_array_adapter<D> f) {
        f.transform_to(c);
    }
    // In order to generate the additional wrapper for Fortran array, the *_WRAPPED_* versions need to be used
    BINDGEN_EXPORT_BINDING_WRAPPED_2(transform_f_to_c, transform_f_to_c_impl<data_store_t>);

    template <class D>
    void transform_c_to_f_impl(fortran_array_adapter<D> f, D c) {
        f.transform_from(c);
    }
    // In order to generate the additional wrapper for Fortran array, the *_WRAPPED_* versions need to be used
    BINDGEN_EXPORT_BINDING_WRAPPED_2(transform_c_to_f, transform_c_to_f_impl<data_store_t>);
} // namespace
