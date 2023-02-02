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
#include <gridtools/stencil/positional.hpp>
#include <gridtools/storage/traits.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    constexpr auto halo = 2;

    struct not_aligned {
        using acc = inout_accessor<0>;
        using out = inout_accessor<1>;
        using i_pos = in_accessor<2>;
        using param_list = make_param_list<acc, out, i_pos>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            auto *ptr = &eval(acc());
            eval(out()) = eval(i_pos()) == halo &&
                          reinterpret_cast<ptrdiff_t>(ptr) % storage::traits::byte_alignment<storage_traits_t>;
        }
    };

    GT_REGRESSION_TEST(alignment_test, test_environment<halo>, stencil_backend_t) {
        auto out = TypeParam::template make_storage<bool>();
        run_single_stage(not_aligned(),
            stencil_backend_t(),
            TypeParam::make_grid(),
            TypeParam::make_storage(),
            out,
            positional<dim::i>());
        TypeParam::verify(false, out);
    }
} // namespace
