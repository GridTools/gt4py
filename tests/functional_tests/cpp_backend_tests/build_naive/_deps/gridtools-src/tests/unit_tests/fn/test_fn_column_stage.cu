/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/column_stage.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        struct sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return tuple(get<0>(acc) + *iter, get<1>(acc) * *iter); },
                    [](auto acc) { return get<0>(acc); });
            }
        };
        struct make_iterator_mock {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const & /*strides*/) {
                    return host_device::at_key<decltype(tag)>(ptr);
                };
            }
        };

        struct device_fun {
            template <class Ptr, class Strides>
            GT_FUNCTION auto operator()(Ptr ptr, Strides strides) const {
                using vdim_t = integral_constant<int, 0>;
                return column_stage<vdim_t, sum_scan, 0, 1>()(tuple(42, 1), 5, make_iterator_mock()(), ptr, strides);
            }
        };

        TEST(scan, device) {
            auto a = cuda_util::cuda_malloc<int>(5);
            auto b = cuda_util::cuda_malloc<int>(5);
            int bh[5] = {1, 2, 3, 4, 5};
            cudaMemcpy(b.get(), bh, 5 * sizeof(int), cudaMemcpyHostToDevice);
            auto composite = sid::composite::keys<integral_constant<int, 0>, integral_constant<int, 1>>::make_values(
                sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(a.get()))
                    .set<property::strides>(tuple(1_c)),
                sid::synthetic()
                    .set<property::origin>(sid::host_device::simple_ptr_holder(b.get()))
                    .set<property::strides>(tuple(1_c)));
            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);
            auto res = on_device::exec(device_fun(), ptr, strides);
            EXPECT_EQ(get<0>(res), 57);
            EXPECT_EQ(get<1>(res), 120);
        }
    } // namespace
} // namespace gridtools::fn
