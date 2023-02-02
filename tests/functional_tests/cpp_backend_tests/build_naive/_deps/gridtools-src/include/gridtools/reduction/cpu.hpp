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

#include <algorithm>
#include <type_traits>

#include "../common/defs.hpp"
#include "functions.hpp"

namespace gridtools {
    namespace reduction {
        struct cpu {};

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
        T reduction_reduce(cpu, T res, plus, T const *buff, size_t n) {
#pragma omp parallel for reduction(+ : res)
            for (size_t i = 0; i < n; i++)
                res += buff[i];
            return res;
        }

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
        T reduction_reduce(cpu, T res, mul, T const *buff, size_t n) {
#pragma omp parallel for reduction(* : res)
            for (size_t i = 0; i < n; i++)
                res *= buff[i];
            return res;
        }

        template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
        T reduction_reduce(cpu, T res, bitwise_and, T const *buff, size_t n) {
#pragma omp parallel for reduction(& : res)
            for (size_t i = 0; i < n; i++)
                res &= buff[i];
            return res;
        }

        template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
        T reduction_reduce(cpu, T res, bitwise_or, T const *buff, size_t n) {
#pragma omp parallel for reduction(| : res)
            for (size_t i = 0; i < n; i++)
                res |= buff[i];
            return res;
        }

        template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
        T reduction_reduce(cpu, T res, bitwise_xor, T const *buff, size_t n) {
#pragma omp parallel for reduction(^ : res)
            for (size_t i = 0; i < n; i++)
                res ^= buff[i];
            return res;
        }

        template <class F, class T>
        T reduction_reduce(cpu, T res, F, T const *buff, size_t n) {
            static_assert(std::is_empty<F>(), "OpenMP reduction supports only stateless functors.");
            static_assert(
                std::is_default_constructible<F>(), "OpenMP reduction supports only default constructible functors.");
#pragma omp declare reduction(gridtools_generic:T : omp_out = F()(omp_out, omp_in)) initializer(omp_priv = omp_orig)
#pragma omp parallel for reduction(gridtools_generic : res)
            for (size_t i = 0; i < n; i++)
                res = F()(res, buff[i]);
            return res;
        }

        inline size_t reduction_round_size(cpu, size_t size) { return size; }
        inline size_t reduction_allocation_size(cpu, size_t size) { return size; }

        template <class T>
        void reduction_fill(cpu, T const &val, T *ptr, size_t data_size, size_t rounded_size, bool has_holes) {
            if (!has_holes) {
                ptr += data_size;
                rounded_size -= data_size;
            }
            std::fill(ptr, ptr + rounded_size, val);
        }
    } // namespace reduction
} // namespace gridtools
