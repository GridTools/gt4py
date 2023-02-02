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

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            eval(out()) = eval(in());
        }
    };

    GT_REGRESSION_TEST(copy_stencil_icosahedral, icosahedral_test_environment<>, stencil_backend_t) {
        auto in = [](int_t i, int_t j, int_t k, int_t c) { return i + j + k + c; };
        auto out = TypeParam::icosahedral_make_storage(cells());
        run_single_stage(functor_copy(),
            stencil_backend_t(),
            TypeParam::make_grid(),
            out,
            TypeParam::icosahedral_make_storage(cells(), in));
        TypeParam::verify(in, out);
    }
} // namespace
