/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/icosahedral.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace icosahedral;

    struct functor_copy {
        using out = inout_accessor<0, cells>;
        using in = in_accessor<1, cells>;
        using param_list = make_param_list<out, in>;
        using location = cells;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(in());
        }
    };

    GT_REGRESSION_TEST(expandable_parameters_icosahedral, icosahedral_test_environment<>, stencil_backend_t) {
        using storages_t = std::vector<decltype(TypeParam::icosahedral_make_storage(cells()))>;
        storages_t out = {TypeParam::icosahedral_make_storage(cells()),
            TypeParam::icosahedral_make_storage(cells()),
            TypeParam::icosahedral_make_storage(cells()),
            TypeParam::icosahedral_make_storage(cells()),
            TypeParam::icosahedral_make_storage(cells())};
        expandable_run<2>([](auto out, auto in) { return execute_parallel().stage(functor_copy(), out, in); },
            stencil_backend_t(),
            TypeParam::make_grid(),
            out,
            storages_t{TypeParam::icosahedral_make_storage(cells(), 10),
                TypeParam::icosahedral_make_storage(cells(), 20),
                TypeParam::icosahedral_make_storage(cells(), 30),
                TypeParam::icosahedral_make_storage(cells(), 40),
                TypeParam::icosahedral_make_storage(cells(), 50)});
        for (size_t i = 0; i != out.size(); ++i)
            TypeParam::verify((i + 1) * 10, out[i]);
    }
} // namespace
