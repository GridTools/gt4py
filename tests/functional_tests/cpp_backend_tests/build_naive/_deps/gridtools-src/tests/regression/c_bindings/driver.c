/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>

#include "implementation.h"

#define I 9
#define J 10
#define K 11

double initial_value(int i, int j, int k) { return i + j + k; }

void init_in(double arr[I][J][K]) {
    int i, j, k;
    for (i = 0; i != I; ++i)
        for (j = 0; j != J; ++j)
            for (k = 0; k != K; ++k)
                arr[i][j][k] = initial_value(i, j, k);
}

void verify(const char *label, double arr[I][J][K]) {
    int i, j, k;
    for (i = 0; i != I; ++i)
        for (j = 0; j != J; ++j)
            for (k = 0; k != K; ++k)
                if (arr[i][j][k] != initial_value(i, j, k)) {
                    fprintf(stderr,
                        "data mismatch in %s[%d][%d][%d]: actual - %f , expected - %f\n",
                        label,
                        i,
                        j,
                        k,
                        arr[i][j][k],
                        initial_value(i, j, k));
                    exit(i);
                }
}

int main() {
    double in[I][J][K];
    double out[I][J][K];
    bindgen_handle *in_handle = create_data_store(I, J, K);
    bindgen_handle *out_handle = create_data_store(I, J, K);

    init_in(in);
    copy_to_data_store(in_handle, (double *)in);

    run_copy_stencil(in_handle, out_handle);

    copy_from_data_store(out_handle, (double *)out);

    bindgen_release(in_handle);
    bindgen_release(out_handle);

    verify("in", in);
    verify("out", out);

    printf("It works!\n");
}
