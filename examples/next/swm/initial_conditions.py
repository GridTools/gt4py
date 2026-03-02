# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize_interior(M, N, dx, dy, a):
    pi = 4.0 * np.arctan(1.0)
    tpi = 2.0 * pi
    d_i = tpi / M
    d_j = tpi / N
    el = N * dx
    pcf = (pi * pi * a * a) / (el * el)

    psi = (
        a
        * np.sin((np.arange(0, M + 1)[:, np.newaxis] + 0.5) * d_i)
        * np.sin((np.arange(0, N + 1) + 0.5) * d_j)
    )
    p = (
        pcf
        * (np.cos(2.0 * np.arange(0, M)[:, np.newaxis] * d_i) + np.cos(2.0 * np.arange(0, N) * d_j))
        + 50000.0
    )

    u = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    return u, v, p


def initialize(M, N, dx, dy, a):
    u, v, p = initialize_interior(M, N, dx, dy, a)

    return (
        np.pad(u, ((1, 0), (0, 1)), mode="wrap"),
        np.pad(v, ((0, 1), (1, 0)), mode="wrap"),
        np.pad(p, ((0, 1), (0, 1)), mode="wrap"),
    )


def initialize_2halo(M, N, dx, dy, a):
    u, v, p = initialize_interior(M, N, dx, dy, a)

    return (
        np.pad(u, ((1, 1), (1, 1)), mode="wrap"),
        np.pad(v, ((1, 1), (1, 1)), mode="wrap"),
        np.pad(p, ((1, 1), (1, 1)), mode="wrap"),
    )
