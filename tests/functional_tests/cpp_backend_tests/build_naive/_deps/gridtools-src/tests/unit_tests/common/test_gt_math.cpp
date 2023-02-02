/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/gt_math.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/host_device.hpp>

#include <verifier.hpp>

using namespace gridtools;

template <class Value>
bool GT_FUNCTION test_pow(Value val, Value result) {
    return expect_with_threshold(math::pow(val, val), result);
}

template <typename Value>
bool GT_FUNCTION test_log(Value val, Value result) {
    return expect_with_threshold(math::log(val), result);
}

template <typename Value>
bool GT_FUNCTION test_exp(Value val, Value result) {
    return expect_with_threshold(math::exp(val), result);
}

static_assert(std::is_same_v<decltype(math::fabs(4.0f)), float>);
static_assert(std::is_same_v<decltype(math::fabs(4.0)), double>);
#ifndef __CUDA_ARCH__
static_assert(std::is_same_v<decltype(math::fabs((long double)4)), long double>);
#endif
static_assert(std::is_same_v<decltype(math::fabs((int)4)), double>);

bool GT_FUNCTION test_fabs() {
    if (!expect_with_threshold(math::fabs(5.6), 5.6, 1e-14))
        return false;
    else if (!expect_with_threshold(math::fabs(-5.6), 5.6, 1e-14))
        return false;
    else if (!expect_with_threshold(math::fabs(-5.6f), 5.6f, 1e-14))
        return false;
    else if (!expect_with_threshold(math::fabs(-5), (double)5, 1e-14))
        return false;
#ifndef __CUDA_ARCH__
    else if (!expect_with_threshold(math::fabs((long double)-5), (long double)5., 1e-14))
        return false;
#endif
    else
        return true;
}

static_assert(std::is_same_v<decltype(math::abs(4.0f)), float>);
static_assert(std::is_same_v<decltype(math::abs(4.0)), double>);
#ifndef __CUDA_ARCH__
static_assert(std::is_same_v<decltype(math::abs((long double)4)), long double>);
#endif

// int overloads
static_assert(std::is_same_v<decltype(math::abs((int)4)), int>);
static_assert(std::is_same_v<decltype(math::abs((long)4)), long>);
static_assert(std::is_same_v<decltype(math::abs((long long)4)), long long>);

GT_FUNCTION bool test_abs() {
    // float overloads
    if (math::abs(5.6) != 5.6)
        return false;
    else if (math::abs(-5.6) != 5.6)
        return false;
    else if (math::abs(-5.6f) != 5.6f)
        return false;
    else if (math::abs(-5) != 5)
        return false;
    else
        return true;
}

namespace {
    TEST(math, test_min) {
        EXPECT_TRUE(math::min(5, 2, 7) == 2);
        EXPECT_TRUE(math::min(5, -1) == -1);

        ASSERT_EQ(math::min(5.3, 22.0, 7.7), 5.3);
    }

    TEST(math, test_min_ref) {
        // checking returned by const &
        double a = 3.5;
        double b = 2.3;
        double const &min = math::min(a, b);
        ASSERT_EQ(min, 2.3);
        b = 8;
        ASSERT_EQ(min, 8);
    }

    TEST(math, test_max) {
        EXPECT_TRUE(math::max(5, 2, 7) == 7);
        EXPECT_TRUE(math::max(5, -1) == 5);

        ASSERT_EQ(math::max(5.3, 22.0, 7.7), 22.0);
    }

    TEST(math, test_max_ref) {
        // checking returned by const &
        double a = 3.5;
        double b = 2.3;
        double const &max = math::max(a, b);

        ASSERT_EQ(max, 3.5);
        a = 8;
        ASSERT_EQ(max, 8);
    }

    TEST(math, test_fabs) { EXPECT_TRUE(test_fabs()); }

    TEST(math, test_abs) { EXPECT_TRUE(test_abs()); }

    TEST(math, test_log) {
        EXPECT_TRUE(test_log<double>(2.3, std::log(2.3)));
        EXPECT_TRUE(test_log<float>(2.3f, std::log(2.3f)));
    }

    TEST(math, test_exp) {
        EXPECT_TRUE(test_exp<double>(2.3, std::exp(2.3)));
        EXPECT_TRUE(test_exp<float>(2.3f, std::exp(2.3f)));
    }

    TEST(math, test_pow) {
        EXPECT_TRUE(test_pow<double>(2.3, std::pow(2.3, 2.3)));
        EXPECT_TRUE(test_pow<float>(2.3f, std::pow(2.3f, 2.3f)));
    }

    TEST(math, test_fmod) {
        EXPECT_FLOAT_EQ(math::fmod(3.7f, 1.2f), std::fmod(3.7f, 1.2f));
        EXPECT_DOUBLE_EQ(math::fmod(3.7, 1.2), std::fmod(3.7, 1.2));
        EXPECT_DOUBLE_EQ(math::fmod(3.7l, 1.2l), std::fmod(3.7l, 1.2l));
    }

    TEST(math, test_trunc) {
        EXPECT_FLOAT_EQ(math::trunc(3.7f), std::trunc(3.7f));
        EXPECT_DOUBLE_EQ(math::trunc(3.7), std::trunc(3.7));
        EXPECT_DOUBLE_EQ(math::trunc(3.7l), std::trunc(3.7l));
    }
} // namespace
