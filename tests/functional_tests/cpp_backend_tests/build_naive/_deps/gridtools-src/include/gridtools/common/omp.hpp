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

#ifdef _OPENMP
#include <omp.h>
#else
extern "C" {
inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
inline double omp_get_wtime() { return 0; }
}
#endif
