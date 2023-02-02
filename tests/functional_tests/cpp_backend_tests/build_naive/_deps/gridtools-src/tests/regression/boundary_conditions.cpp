/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/boundaries/boundary.hpp>

#include <gcl_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace boundaries;

    template <class Float>
    constexpr Float face_value = 88;
    template <class Float>
    constexpr Float edge_value = 77777;
    template <class Float>
    constexpr Float corner_value = 55555;
    template <class Float>
    constexpr Float factor = 2;

    template <class Float>
    struct direction_bc_input {
        // relative coordinates
        template <typename Direction, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(
            Direction, DataField0 &data_field0, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field1(i, j, k) = data_field0(i, j, k) * factor<Float>;
        }

        // relative coordinates
        template <sign I, sign K, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(
            direction<I, minus_, K>, DataField0 &, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field1(i, j, k) = face_value<Float> * factor<Float>;
        }

        // relative coordinates
        template <sign K, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(direction<minus_, minus_, K>,
            DataField0 &,
            DataField1 const &data_field1,
            uint_t i,
            uint_t j,
            uint_t k) const {
            data_field1(i, j, k) = edge_value<Float> * factor<Float>;
        }

        template <typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(direction<minus_, minus_, minus_>,
            DataField0 &,
            DataField1 const &data_field1,
            uint_t i,
            uint_t j,
            uint_t k) const {
            data_field1(i, j, k) = corner_value<Float> * factor<Float>;
        }
    };

    template <typename T>
    void verify_result(array<halo_descriptor, 3> const &halos, T &&src, T &&dst) {
        auto src_v = src->const_host_view();
        auto dst_v = dst->host_view();
        using float_t = std::decay_t<decltype(src_v(0, 0, 0))>;

        // check inner domain (should be zero)
        for (uint_t i = halos[0].begin(); i <= halos[0].end(); ++i)
            for (uint_t j = halos[1].begin(); j <= halos[1].end(); ++j)
                for (uint_t k = halos[2].begin(); k <= halos[2].end(); ++k) {
                    EXPECT_EQ(src_v(i, j, k), i + j + k);
                    EXPECT_EQ(dst_v(i, j, k), 0);
                    dst_v(i, j, k) = -1;
                }

        // check corner (direction<minus_, minus_, minus_>)
        for (uint_t i = 0; i < halos[0].begin(); ++i)
            for (uint_t j = 0; j < halos[1].begin(); ++j)
                for (uint_t k = 0; k < halos[2].begin(); ++k) {
                    EXPECT_EQ(dst_v(i, j, k), factor<float_t> * corner_value<float_t>);
                    dst_v(i, j, k) = -1;
                }

        // check edge (direction<minus_, minus_, K>)
        for (uint_t i = 0; i < halos[0].begin(); ++i)
            for (uint_t j = 0; j < halos[1].begin(); ++j)
                for (uint_t k = halos[2].begin(); k <= halos[2].end() + halos[2].plus(); ++k) {
                    EXPECT_EQ(dst_v(i, j, k), factor<float_t> * edge_value<float_t>);
                    dst_v(i, j, k) = -1;
                }

        // check face (direction<I, minus_, K>)
        for (uint_t i = halos[0].begin(); i <= halos[0].end() + halos[0].plus(); ++i)
            for (uint_t j = 0; j < halos[1].begin(); ++j)
                for (uint_t k = 0; k < halos[2].end() + halos[2].plus(); ++k) {
                    EXPECT_EQ(dst_v(i, j, k), factor<float_t> * face_value<float_t>);
                    dst_v(i, j, k) = -1;
                }

        // remainder
        for (uint_t i = 0; i < halos[0].end() + halos[0].plus(); ++i)
            for (uint_t j = halos[1].begin(); j < halos[1].end() + halos[1].plus(); ++j)
                for (uint_t k = 0; k < halos[2].end() + halos[2].plus(); ++k)
                    if (i < halos[0].begin() || i > halos[0].end() || k < halos[2].begin() || k > halos[2].end() ||
                        j > halos[1].end()) {
                        EXPECT_EQ(dst_v(i, j, k), factor<float_t> * src_v(i, j, k));
                        dst_v(i, j, k) = -1;
                    }

        // test the test (all values should be set to -1 now)
        for (uint_t i = 0; i < halos[0].end() + halos[0].plus(); ++i)
            for (uint_t j = 0; j < halos[1].end() + halos[1].plus(); ++j)
                for (uint_t k = 0; k < halos[2].end() + halos[2].plus(); ++k)
                    ASSERT_EQ(dst_v(i, j, k), -1);
    }

    constexpr auto halo_size = 3;

    GT_REGRESSION_TEST(distributed_boundary, test_environment<halo_size>, gcl_arch_t) {
        auto src = TypeParam::make_storage([](int i, int j, int k) { return i + j + k; });
        auto dst = TypeParam::make_storage(0);

        auto &&lengths = src->info().lengths();
        auto &&total_lengths = make_total_lengths(*src);
        array<halo_descriptor, 3> halos;
        for (size_t i = 0; i != 3; ++i)
            halos[i] = {halo_size, halo_size, halo_size, lengths[i] - halo_size - 1, total_lengths[i]};

        auto testee = [&] {
            make_boundary<gcl_arch_t>(halos, direction_bc_input<typename TypeParam::float_t>()).apply(src, dst);
        };

        testee();
        verify_result(halos, src, dst);

        TypeParam::benchmark("distributed_boundary", testee);
    }
} // namespace
