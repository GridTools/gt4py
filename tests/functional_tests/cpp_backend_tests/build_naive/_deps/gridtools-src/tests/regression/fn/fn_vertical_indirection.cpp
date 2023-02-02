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

#include <gridtools/fn/unstructured.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    struct derivative_stencil {
        GT_FUNCTION auto operator()() const {
            return [](auto const &field, auto const &offset) {
                const auto shifted = shift(field, unstructured::dim::vertical{}, deref(offset));
                return deref(field) - deref(shifted);
            };
        }
    };

    constexpr auto parabolic = [](auto /* vertex */, auto k) -> float {
        const float k_shifted = float(k) + 0.5f;
        return 0.5f * k_shifted * k_shifted;
    };
    constexpr auto offsets = [](auto /* vertex */, auto k) -> int { return k > 0 ? -1 : 0; };
    constexpr auto linear = [](auto /* vertex */, auto k) { return k; };

    constexpr auto compute_derivative = [](auto executor, auto const &input, auto const &offsets, auto &output) {
        executor().arg(input).arg(offsets).arg(output).assign(2_c, derivative_stencil(), 0_c, 1_c).execute();
    };

    GT_REGRESSION_TEST(fn_vertical_indirection, test_environment<>, fn_backend_t) {
        auto fencil = [](int nvertices, int nlevels, auto const &field, auto &offets, auto &output) {
            auto be = fn_backend_t();
            auto domain = unstructured_domain({nvertices, nlevels}, tuple{0, 0});
            auto backend = make_backend(be, domain);
            compute_derivative(backend.stencil_executor(), field, offets, output);
        };

        auto mesh = TypeParam::fn_unstructured_mesh();
        const auto input_offsets = mesh.template make_const_storage<int>(offsets, mesh.nvertices(), mesh.nlevels());
        const auto input_parabola = mesh.make_const_storage(parabolic, mesh.nvertices(), mesh.nlevels());
        auto output_derivative = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
        fencil(mesh.nvertices(), mesh.nlevels(), input_parabola, input_offsets, output_derivative);
        TypeParam::verify(linear, output_derivative);
    }
} // namespace
