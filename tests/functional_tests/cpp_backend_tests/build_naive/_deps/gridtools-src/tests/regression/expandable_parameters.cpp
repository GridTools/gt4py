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

    struct copy_functor {
        using parameters_out = inout_accessor<0>;
        using parameters_in = accessor<1>;

        using param_list = make_param_list<parameters_out, parameters_in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(parameters_out{}) = eval(parameters_in{});
        }
    };

    GT_REGRESSION_TEST(expandable_parameters, test_environment<>, stencil_backend_t) {
        using storages_t = std::vector<typename TypeParam::storage_type>;
        storages_t out = {TypeParam::make_storage(1.),
            TypeParam::make_storage(2.),
            TypeParam::make_storage(3.),
            TypeParam::make_storage(4.),
            TypeParam::make_storage(5.)};
        storages_t in = {TypeParam::make_storage(-1.),
            TypeParam::make_storage(-2.),
            TypeParam::make_storage(-3.),
            TypeParam::make_storage(-4.),
            TypeParam::make_storage(-5.)};

        expandable_run<2>(
            [](auto in, auto out) {
                GT_DECLARE_EXPANDABLE_TMP(typename TypeParam::float_t, tmp);
                return execute_parallel().ij_cached(tmp).stage(copy_functor(), tmp, in).stage(copy_functor(), out, tmp);
            },
            stencil_backend_t(),
            TypeParam::make_grid(),
            in,
            out);

        for (size_t i = 0; i != in.size(); ++i)
            TypeParam::verify(in[i], out[i]);
    }
} // namespace
