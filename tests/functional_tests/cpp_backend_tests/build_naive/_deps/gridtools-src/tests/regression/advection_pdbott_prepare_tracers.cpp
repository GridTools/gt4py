/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <vector>

#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct prepare_tracers {
        using data = inout_accessor<0>;
        using data_nnow = in_accessor<1>;
        using rho = in_accessor<2>;

        using param_list = make_param_list<data, data_nnow, rho>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            eval(data()) = eval(rho()) * eval(data_nnow());
        }
    };

    GT_REGRESSION_TEST(advection_pdbott_prepare_tracers, test_environment<>, stencil_backend_t) {
        std::vector<typename TypeParam::storage_type> in, out;

        for (size_t i = 0; i < 11; ++i) {
            out.push_back(TypeParam::make_storage());
            in.push_back(TypeParam::make_storage(i));
        }

        auto comp = [&, grid = TypeParam::make_grid(), rho = TypeParam::make_const_storage(1.1)] {
            expandable_run<2>(
                [](auto out, auto in, auto rho) { return execute_parallel().stage(prepare_tracers(), out, in, rho); },
                stencil_backend_t(),
                grid,
                out,
                in,
                rho);
        };

        comp();
        for (size_t i = 0; i != out.size(); ++i)
            TypeParam::verify([i](int, int, int) { return 1.1 * i; }, out[i]);

        TypeParam::benchmark("advection_pdbott_prepare_tracers", comp);
    }
} // namespace
