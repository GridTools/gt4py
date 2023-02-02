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
namespace gridtools {

    /** \ingroup common
        @{
        \ingroup atomic
        @{
    */

    template <typename T>
    class atomic_host {

      public:
        /**
         * Function computing an atomic addition
         * @param var reference to variable where the addition is performed
         * @param val value added to var
         * @return the old value contained in var
         */
        GT_FUNCTION
        static T atomic_add(T &var, const T val) {
#if _OPENMP > 201106
            T old;
#pragma omp atomic capture
            {
                old = var;
                var += val;
            }
            return old;
#else
            T old;
#pragma omp critical(AtomicAdd)
            {
                old = var;
                var += val;
            }
            return old;
#endif
        }

        /**
         * Function computing an atomic substraction
         * @param var reference to variable where the substraction is performed
         * @param val value added to var
         * @return the old value contained in var
         */
        GT_FUNCTION
        static T atomic_sub(T &var, const T val) {
#if _OPENMP > 201106
            T old;
#pragma omp atomic capture
            {
                old = var;
                var -= val;
            }
            return old;
#else
            T old;
#pragma omp critical(AtomicSub)
            {
                old = var;
                var -= val;
            }
            return old;
#endif
        }

        /**
         * Function computing an atomic exchange of value of a variable
         * @param var reference to variable which value is replaced by val
         * @param val value inserted in variable var
         * @return the old value contained in var
         */
        GT_FUNCTION
        static T atomic_exch(T &var, const T val) {
#if _OPENMP > 201106
            T old;
#pragma omp atomic capture
            {
                old = var;
                var = val;
            }
            return old;
#else
            T old;
#pragma omp critical(exch)
            {
                old = var;
                var = val;
            }
            return old;
#endif
        }

        /**
         * Function computing an atomic min operation
         * @param var reference used to compute and store the min
         * @param val value used in the min comparison
         * @return the old value contained in var
         */
        GT_FUNCTION
        static T atomic_min(T &var, const T val) {
            T old;
#pragma omp critical(min)
            {
                old = var;
                var = std::min(var, val);
            }
            return old;
        }

        /**
         * Function computing an atomic max operation
         * @param var reference used to compute and store the min
         * @param val value used in the min comparison
         * @return the old value contained in var
         */
        GT_FUNCTION
        static T atomic_max(T &var, const T val) {
            T old;
#pragma omp critical(max)
            {
                old = var;
                var = std::max(var, val);
            }
            return old;
        }
    };

    /** @} */
    /** @} */

} // namespace gridtools
