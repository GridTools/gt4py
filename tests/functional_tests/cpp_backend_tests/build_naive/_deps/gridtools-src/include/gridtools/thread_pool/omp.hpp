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

#if defined(_OPENMP) || defined(GT_HIP_OPENMP_WORKAROUND)
#include <omp.h>
#endif

namespace gridtools {
    namespace thread_pool {
        struct omp {
#if defined(_OPENMP) || defined(GT_HIP_OPENMP_WORKAROUND)
            friend auto thread_pool_get_thread_num(omp) { return omp_get_thread_num(); }
            friend auto thread_pool_get_max_threads(omp) { return omp_get_max_threads(); }

            template <class F, class I>
            friend void thread_pool_parallel_for_loop(omp, F const &f, I lim) {
#pragma omp parallel for
                for (I i = 0; i < lim; ++i)
                    f(i);
            }

            template <class F, class I, class J>
            friend void thread_pool_parallel_for_loop(omp, F const &f, I i_lim, J j_lim) {
#pragma omp parallel for collapse(2)
                for (J j = 0; j < j_lim; ++j)
                    for (I i = 0; i < i_lim; ++i)
                        f(i, j);
            }

            template <class F, class I, class J, class K>
            friend void thread_pool_parallel_for_loop(omp, F const &f, I i_lim, J j_lim, K k_lim) {
#pragma omp parallel for collapse(3)
                for (K k = 0; k < k_lim; ++k)
                    for (J j = 0; j < j_lim; ++j)
                        for (I i = 0; i < i_lim; ++i)
                            f(i, j, k);
            }
#endif
        };
    } // namespace thread_pool
} // namespace gridtools
