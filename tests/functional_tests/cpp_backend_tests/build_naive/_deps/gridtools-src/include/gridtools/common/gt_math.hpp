/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cmath>
#include <cstdlib>
#include <type_traits>

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \defgroup math Mathematical Functions
        @{
    */

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <uint_t Number>
    struct gt_pow {
        template <typename Value>
        GT_FUNCTION static Value constexpr apply(Value const &v) {
            return v * gt_pow<Number - 1>::apply(v);
        }
    };

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <>
    struct gt_pow<0> {
        template <typename Value>
        GT_FUNCTION static Value constexpr apply(Value const &) {
            return 1.;
        }
    };

    /*
     * @brief helper function that provides a static version of std::ceil
     * @param num value to ceil
     * @return ceiled value
     */
    GT_FUNCTION constexpr int gt_ceil(float num) {
        int n = static_cast<int>(num);
        return static_cast<float>(n) == num ? n : n + (num > 0 ? 1 : 0);
    }

    namespace math {
        template <typename Value>
        GT_FUNCTION constexpr Value const &max(Value const &val0) {
            return val0;
        }

        template <typename Value>
        GT_FUNCTION constexpr Value const &max(Value const &val0, Value const &val1) {
            return val0 > val1 ? val0 : val1;
        }

        template <typename Value, typename... OtherValues>
        GT_FUNCTION constexpr Value const &max(Value const &val0, Value const &val1, OtherValues const &...vals) {
            return val0 > max(val1, vals...) ? val0 : max(val1, vals...);
        }

        template <typename Value>
        GT_FUNCTION constexpr Value const &min(Value const &val0) {
            return val0;
        }

        template <typename Value>
        GT_FUNCTION constexpr Value const &min(Value const &val0, Value const &val1) {
            return val0 > val1 ? val1 : val0;
        }

        template <typename Value, typename... OtherValues>
        GT_FUNCTION constexpr Value const &min(Value const &val0, Value const &val1, OtherValues const &...vals) {
            return val0 > min(val1, vals...) ? min(val1, vals...) : val0;
        }

#if defined(GT_CUDACC) && defined(__NVCC__)
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) fabs(double val) { return ::fabs(val); }

        GT_FUNCTION decltype(auto) fabs(float val) { return ::fabs(val); }

        template <typename Value>
        GT_FUNCTION decltype(auto) fabs(Value val) {
            return ::fabs((double)val);
        }
        GT_FUNCTION_HOST decltype(auto) fabs(long double val) { return std::fabs(val); }
#else
        using std::fabs;
#endif

#ifdef GT_CUDACC
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) abs(int val) { return ::abs(val); }

        GT_FUNCTION decltype(auto) abs(long val) { return ::abs(val); }

        GT_FUNCTION decltype(auto) abs(long long val) { return ::abs(val); }

        // forward to fabs
        template <typename Value>
        GT_FUNCTION decltype(auto) abs(Value val) {
            return math::fabs(val);
        }
#else
        using std::abs;
#endif

#ifdef __CUDA_ARCH__
        GT_FUNCTION_DEVICE float exp(float x) { return ::expf(x); }

        GT_FUNCTION_DEVICE double exp(double x) { return ::exp(x); }
#else
        using std::exp;
#endif

#ifdef __CUDA_ARCH__
        GT_FUNCTION_DEVICE float log(float x) { return ::logf(x); }

        GT_FUNCTION_DEVICE double log(double x) { return ::log(x); }
#else
        using std::log;
#endif

#ifdef __CUDA_ARCH__
        GT_FUNCTION_DEVICE float pow(float x, float y) { return ::powf(x, y); }

        GT_FUNCTION_DEVICE double pow(double x, double y) { return ::pow(x, y); }
#else
        using std::pow;
#endif

#ifdef __CUDA_ARCH__
        GT_FUNCTION_DEVICE float sqrt(float x) { return ::sqrtf(x); }

        GT_FUNCTION_DEVICE double sqrt(double x) { return ::sqrt(x); }
#else
        using std::sqrt;
#endif

#ifdef GT_CUDACC
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) fmod(float x, float y) { return ::fmodf(x, y); }

        GT_FUNCTION decltype(auto) fmod(double x, double y) { return ::fmod(x, y); }

        GT_FUNCTION_HOST decltype(auto) fmod(long double x, long double y) { return std::fmod(x, y); }
#else
        using std::fmod;
#endif

#ifdef GT_CUDACC
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) trunc(float val) { return ::truncf(val); }

        template <class Value, std::enable_if_t<std::is_convertible_v<Value, double>, int> = 0>
        GT_FUNCTION decltype(auto) trunc(Value val) {
            return ::trunc(val);
        }

        GT_FUNCTION_HOST decltype(auto) trunc(long double val) { return std::trunc(val); }
#else
        using std::trunc;
#endif
    } // namespace math

    /** @} */
    /** @} */
} // namespace gridtools
