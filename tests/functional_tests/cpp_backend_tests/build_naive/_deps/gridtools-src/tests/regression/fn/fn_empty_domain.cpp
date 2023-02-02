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

    struct empty_stencil {
        GT_FUNCTION constexpr auto operator()() const {
            return []() { return 0.0f; };
        }
    };

    GT_REGRESSION_TEST(empty_domain_stencil, test_environment<>, fn_backend_t) {
        auto out = TypeParam::make_storage([](...) { return 0; });

        auto domain = unstructured_domain(tuple{0, 0}, tuple{0, 0});
        auto backend = make_backend(fn_backend_t{}, domain);
        backend.stencil_executor()().arg(out).assign(0_c, empty_stencil{}).execute();
    }

    struct empty_column : fwd {
        static GT_FUNCTION constexpr auto prologue() {
            return tuple(scan_pass([](auto acc) { return acc; }, host_device::identity()));
        }

        static GT_FUNCTION constexpr auto body() {
            return scan_pass([](auto acc) { return acc; }, host_device::identity());
        }
    };

    GT_REGRESSION_TEST(empty_domain_column, vertical_test_environment<>, fn_backend_t) {
        auto out = TypeParam::make_storage([](...) { return 0; });

        auto domain = unstructured_domain(tuple{0, 0}, tuple{0, 0});
        auto backend = make_backend(fn_backend_t{}, domain);
        backend.vertical_executor()().arg(out).assign(0_c, empty_column{}, 0.0f).execute();
    }

} // namespace
