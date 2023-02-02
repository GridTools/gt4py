/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/stencil_stage.hpp>

#include <gtest/gtest.h>

#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        template <int I>
        using int_t = integral_constant<int, I>;

        struct stencil {
            GT_FUNCTION constexpr auto operator()() const {
                return [](auto const &iter) { return 2 * *iter; };
            }
        };

        struct make_iterator_mock {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const & /*strides*/) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(stencil_stage, smoke) {
            int in[1] = {42}, out[1] = {0};

            auto as_synthetic = [](int x[3]) {
                return sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(&x[0]))
                    .set<property::strides>(tuple(1_c));
            };

            auto composite = sid::composite::keys<int_t<0>, int_t<1>>::make_values(as_synthetic(out), as_synthetic(in));

            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);

            stencil_stage<stencil, 0, 1> ss;
            ss(make_iterator_mock()(), ptr, strides);
            EXPECT_EQ(in[0], 42);
            EXPECT_EQ(out[0], 84);

            merged_stencil_stage<stencil_stage<stencil, 1, 0>, stencil_stage<stencil, 0, 1>> mss;
            mss(make_iterator_mock()(), ptr, strides);
            EXPECT_EQ(in[0], 168);
            EXPECT_EQ(out[0], 336);
        }

    } // namespace
} // namespace gridtools::fn
